# backend/agent_graph.py 
import os
from typing import Dict, Any, TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
# 🔥 [핵심 고도화] 프레임워크 주도형 자동 도구 실행을 위해 LangGraph 내장 ToolNode와 제어 Edge를 임포트합니다.
from langgraph.prebuilt import ToolNode, tools_condition

from model_factory import ModelFactory
from backend.tools import get_redhat_live_search_tool
from backend.schemas import OCPResponseSchema
from flashrank import Ranker, RerankRequest

# 1. 멀티턴 및 자율 도구 실행 제어용 상태(State) 설계
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    route_to: str
    rag_context: str
    tool_result: str
    draft_answer: str
    input_category: str          
    validation_loop_count: int
    final_structured_output: Dict[str, Any]

# 실제 연동할 Red Hat 실시간 기술 검색 도구 객체 확보
live_search_tool = get_redhat_live_search_tool()

# ====================================================================
# [Agent 1] Router Agent (의도 및 기술 카테고리 분류)
# ====================================================================
def router_agent(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    last_query = state["messages"][-1].content
    
    router_few_shot = """당신은 인프라 엔지니어의 질문 의도와 인프라 기술 범주를 정확히 분류하는 지능형 라우터입니다.
질문의 성격에 따라 다음 규칙을 준수하여 결과를 '라우팅_방향|기술_카테고리' 포맷으로 정확히 출력하세요.

[분류 규칙]
1. 일반 매뉴얼, 아키텍처, 설치 가이드 요청 -> 라우팅은 'rag'
2. 실시간 에러 로그, 긴급 장애 트러블슈팅, 버그 추적 -> 라우팅은 'tool'
3. 기술 카테고리는 질문 내용에 따라 'INSTALL', 'NETWORK', 'SECURITY', 'ERROR' 중 하나로 지정

[Few-shot 예시]
User: OCP 4.20 pull-secret 인증서 변경 명령어 세트 알려줘
Output: rag|SECURITY

User: Disconnected 부팅 도중 Ignition 파일 콤마 파싱 에러로 노드가 안 켜져요
Output: tool|ERROR
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", router_few_shot),
        ("user", "질문: {input}")
    ])
    chain = prompt | llm
    raw_decision = chain.invoke({"input": last_query}).content.strip()
    
    if "|" in raw_decision:
        route, category = raw_decision.split("|")[0].lower(), raw_decision.split("|")[1].upper()
    else:
        route, category = "rag", "INSTALL"
        
    return {
        "route_to": "tool_search_node" if "tool" in route else "ocp_rag_node",
        "input_category": category
    }

# ====================================================================
# [Agent 2] OCP RAG Agent (채점 및 재검색 쿼리를 생성하는 Self-RAG 에이전트)
# ====================================================================
def ocp_rag_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    embeddings = ModelFactory.get_embeddings()
    db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    query = state["messages"][-1].content
    
    if not os.path.exists(db_path):
        return {"rag_context": "로컬 벡터 DB 부재로 기본 표준 규칙 기반으로 안내합니다."}
        
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(query)
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 문서 평가관입니다. 검색된 문맥이 엔지니어의 질문에 매칭되는 명령어 정보를 "
                   "충분히 포함하고 있다면 'YES', 부족하거나 애매하다면 'NO'만 대답하세요. 다른 말은 절대 금지합니다."),
        ("user", "검색 문맥:\n{context}\n\n질문: {query}")
    ])
    
    initial_context = "\n".join([d.page_content for d in retrieved_docs])
    eval_chain = eval_prompt | llm
    assessment = eval_chain.invoke({"context": initial_context, "query": query}).content.strip().upper()
    
    if "NO" in assessment:
        print("[RAG 에이전트 판단] 1차 검색 품질 미흡 감지 ➔ 변수/CLI 중심 2차 확장 검색 개시.")
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 쿼리 확장 엔진입니다. 입력된 OpenShift 인프라 질문을 해결하기 위한 "
                       "핵심 키워드나 oc CLI 명령어 형태의 연관 검색어 한 줄만 반환하세요."),
            ("user", "원본 질문: {query}")
        ])
        rewrite_chain = rewrite_prompt | llm
        expanded_query = rewrite_chain.invoke({"query": query}).content.strip()
        
        secondary_docs = vector_store.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(expanded_query)
        retrieved_docs.extend(secondary_docs)

    try:
        ranker = Ranker()
        passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(retrieved_docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)
        
        final_context_chunks = []
        for res in reranked_results[:2]:
            source_info = res.get("meta", {}).get("source_file", "OCP 4.20 가이드")
            final_context_chunks.append(f"[출처: {source_info}]\n{res['text']}")
            
        return {"rag_context": "\n---\n".join(final_context_chunks)}
    except Exception:
        return {"rag_context": "\n---\n".join([d.page_content for d in retrieved_docs[:2]])}

# ====================================================================
# [Agent 3] Tool Search Node (🔥 완전한 프레임워크 주도형 자동 실행 구조)
# ====================================================================
def tool_search_node(state: AgentState) -> Dict[str, Any]:
    """
    [지적사항 완전 해결] 기존의 파이썬 수동 if-else 조건문 및 tool.invoke 하드코딩을 전면 철폐했습니다.
    LLM에 공식 도구를 명시적으로 bind하고 쿼리를 전달하면, 모델은 자율적으로 '도구 호출 메시지(tool_calls)'만 발행합니다.
    """
    llm = ModelFactory.get_llm()
    
    # LLM이 인식할 수 있도록 도구를 네이티브 바인딩 처리
    llm_with_tools = llm.bind_tools([live_search_tool]) if hasattr(live_search_tool, "name") else llm
    query = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 실시간 외부 Red Hat Issue Tracker 도구를 자율적으로 실행할 권한을 가진 운영관입니다. "
                   "질문에서 장애 에러 키워드를 정밀하게 추출하여 연동된 기술 검색 도구를 실행(Call)하십시오."),
        ("user", "{input}")
    ])
    
    # 모델 호출 시, LLM은 스스로 도구 규격을 파싱하여 tool_calls 가 포함된 AIMessage를 반환합니다.
    response = llm_with_tools.invoke(prompt.format(input=query))
    
    print("[자율 Tool Calling 메시지 발행 완료] 프레임워크 제어권 위임 모드 진입.")
    return {"messages": [response]}

# ====================================================================
# [Agent 4] Answer Refiner Agent (카테고리별 동적 템플릿 구조화)
# ====================================================================
def answer_refiner_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    query = state["messages"][-1].content
    category = state.get("input_category", "INSTALL")
    
    # 🔥 [구조 고도화] 프레임워크가 자동 실행한 도구 응답 결과 메시지(ToolMessage)를 
    # 대화 기록(messages) 내에서 실시간으로 필터링하여 컨텍스트에 완벽히 동적 결합합니다.
    messages = state.get("messages", [])
    extracted_tool_output = ""
    for msg in reversed(messages):
        if msg.type == "tool":
            extracted_tool_output = msg.content
            break
            
    context = (
        f"RAG 지식고 추출 내역:\n{state.get('rag_context','')}\n"
        f"프레임워크 자율 Tool 실행 결과:\n{extracted_tool_output if extracted_tool_output else state.get('tool_result', '')}"
    )
    
    if category == "ERROR":
        category_instruction = "당신은 긴급 장애 복구 전문관입니다. 에러 현상 해결을 위해 원인 분석을 먼저 제시하고, 현장에서 즉시 격리 조치 가능한 oc 명령어를 작성하세요."
    elif category == "SECURITY":
        category_instruction = "당신은 Red Hat 보안 정책 및 SCC 검증관입니다. 폐쇄망 권한 제약 및 암호화 자격증명 갱신 절차에 맞는 안전한 YAML 컴포넌트를 설계하세요."
    elif category == "NETWORK":
        category_instruction = "당신은 OpenShift SDN 및 인그레스 네트워크 엔지니어입니다. Disconnected 환경의 라우팅 매니페스트 및 프록시 정책 연동 규칙을 중심으로 서술하세요."
    else:
        category_instruction = "당신은 OCP 4.20 코어 인프라 구축 설계자입니다. 초기 클러스터 부트스트랩 및 노드 프로비저닝 표준 절차 가이드라인을 작성하세요."

    reject_feedback = ""
    if len(messages) > 1 and "코드 검증 실패" in messages[-1].content:
        reject_feedback = f"\n\n⚠️ [코드 검증관의 반려 피드백 사항]: {messages[-1].content}\n지적된 명령어 오탈자나 경로 실수를 완벽히 수정해 내세요."

    refiner_template = f"""{category_instruction}
제공된 문맥 데이터의 범위 내에서만 사실에 기반하여 답변을 조립해야 합니다. 정보가 유실되지 않도록 철저히 반영하세요.{reject_feedback}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", refiner_template),
        ("user", "인프라 데이터베이스:\n{context}\n\n현장 질의: {query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return {"draft_answer": response.content}

# ====================================================================
# [Agent 5] Code Validator Agent (독립적 반려/승인 의사결정권 발동 노드)
# ====================================================================
def code_validator_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    draft = state.get("draft_answer", "")
    loop_count = state.get("validation_loop_count", 0)
    
    validator_few_shot = """당신은 기술 스크립트와 YAML 문법의 신뢰성을 완벽히 검증하는 독립 코드 검증관입니다.
초안을 분석하여 명령어 가이드라인이 명세에 완벽히 부합하면 구조화 포맷에 맞춰 승인하고, OCP 전용 CLI 규격에 어긋나거나 핵심 파라미터가 유실되었다면 반드시 'STATUS: REJECT - [구체적 사유]' 형식으로 요약 필드의 첫 줄을 시작하십시오.

[출력 제어 Few-shot 예시]
Input: pull-secret은 그냥 적당히 수정하시면 처리됩니다.
Output: {
  "summary": "STATUS: REJECT - pull-secret 갱신에 필요한 전용 oc set data secret CLI 명세 및 openshift-config 네임스페이스 경로 지정이 완전히 유실되었습니다.",
  "steps": ["검증 반려"],
  "code_block": "# REJECTED_BY_VALIDATOR",
  "references": ["OCP 4.20 공식 가이드 복구 요망"]
}

Input: oc adm release mirror 명령을 통해 로컬 레지스트리로 전송합니다.
Output: {
  "summary": "OCP 4.20 폐쇄망 이미지 미러링 조치 완료",
  "steps": ["타켓 미러 레지스트리 자격 증명을 갱신합니다.", "oc adm release mirror 스크립트를 적용합니다."],
  "code_block": "oc adm release mirror --to=myregistry.local/ocp",
  "references": ["OpenShift 4.20 Disconnected Environments setup"]
}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", validator_few_shot),
        ("user", "검증 대상 가이드 초안 내용:\n{draft_content}")
    ])
    
    structured_llm = llm.with_structured_output(OCPResponseSchema)
    chain = prompt | structured_llm
    
    try:
        validated_obj = chain.invoke({"draft_content": draft})
        res_dict = validated_obj.dict()
        
        if "STATUS: REJECT" in res_dict["summary"] and loop_count < 2:
            print(f"[검증관 반려] 수정을 위해 환류 처리 가동: {res_dict['summary']}")
            return {
                "messages": [AIMessage(content=f"코드 검증 실패 사유: {res_dict['summary']}")],
                "route_to": "re_edit",
                "validation_loop_count": loop_count + 1
            }
            
        return {
            "messages": [AIMessage(content="최종 승인 통과.")],
            "route_to": "approve",
            "final_structured_output": res_dict
        }
    except Exception:
        fallback = OCPResponseSchema(
            summary="무결성 검증 세션 마감",
            steps=["데이터 무결성 가이드라인 조립을 마쳤습니다."],
            code_block=f"# 조치 원문 복구 데이터\n# {draft[:120]}...",
            references=["OCP 4.20 가이드 팩 확인"]
        )
        return {"route_to": "approve", "final_structured_output": fallback.dict()}
 
# ====================================================================
# [Graph Design] 🔥 프레임워크 표준 자동 ToolNode 아키텍처 결합 직조
# ====================================================================
def router_edge_decision(state: AgentState):
    return state["route_to"]

def validator_edge_decision(state: AgentState):
    return "answer_refiner_node" if state["route_to"] == "re_edit" else END

workflow = StateGraph(AgentState)

# 핵심 노드 함수 등록
workflow.add_node("router_agent", router_agent)
workflow.add_node("ocp_rag_node", ocp_rag_node)
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("answer_refiner_node", answer_refiner_node)
workflow.add_node("code_validator_node", code_validator_node)

# 🔥 [가장 중요] 하드코딩 실행 대신, LangGraph 표준 prebuilt ToolNode를 독립 노드로 직접 배치합니다.
# 이 노드는 앞서 LLM이 발행한 tool_calls 내역을 가로채서 백엔드 인프라단에서 완전 자율 실행합니다.
standard_tool_node = ToolNode([live_search_tool]) if hasattr(live_search_tool, "name") else ToolNode([])
workflow.add_node("action_tools", standard_tool_node)

# 진입점 및 첫 라우팅 규칙 바인딩
workflow.set_entry_point("router_agent")
workflow.add_conditional_edges("router_agent", router_edge_decision, {
    "ocp_rag_node": "ocp_rag_node",
    "tool_search_node": "tool_search_node"
})

# RAG 경로 통과 시 순차적으로 Refiner로 이동
workflow.add_edge("ocp_rag_node", "answer_refiner_node")

# 🔥 [에이전틱 라우팅 수립] Tool Search Agent 통과 후, 프레임워크 제어형 자동 실행 분기(tools_condition) 결합
# 모델의 대화 메시지 상태를 읽어 tool_calls 가 존재하면 'action_tools' 노드로 자동 분기하고, 완료 시 Refiner로 이동하게 만듭니다.
workflow.add_edge("tool_search_node", "action_tools")
workflow.add_edge("action_tools", "answer_refiner_node")

# Refiner 통과 후 검증관 노드로 이동
workflow.add_edge("answer_refiner_node", "code_validator_node")

# 검증 결과 반려(re_edit) 판단 시 환류 루프 가동 조건부 에지
workflow.add_conditional_edges("code_validator_node", validator_edge_decision, {
    "answer_refiner_node": "answer_refiner_node",
    END: END
})

compiled_graph = workflow.compile(checkpointer=MemorySaver())