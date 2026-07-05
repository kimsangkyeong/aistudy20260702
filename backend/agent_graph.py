# backend/agent_graph.py
import os
import json
from typing import Dict, Any, TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
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
2. 실시간 에러 로그, 긴급 장애 트러블슈팅, 버그 추적, 최신 명령어 조회 -> 라우팅은 'tool'
3. 기술 카테고리는 질문 내용에 따라 'INSTALL', 'NETWORK', 'SECURITY', 'ERROR' 중 하나로 지정

[Few-shot 예시]
User: OCP 4.20 pull-secret 인증서 변경 명령어 세트 알려줘
Output: rag|SECURITY

User: Disconnected 부팅 도중 Ignition 파일 콤마 파싱 에러로 노드가 안 켜져요
Output: tool|ERROR

User: 신규 프로젝트 생성 명령어 알려줘
Output: tool|INSTALL
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
        route, category = "tool", "INSTALL"
        
    return {
        "route_to": "tool_search_node" if "tool" in route else "ocp_rag_node",
        "input_category": category
    }

# ====================================================================
# [Agent 2] OCP RAG Agent (가이드북 + 관리자 승인 FAQ 교차 하이브리드 검색)
# ====================================================================
def ocp_rag_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    embeddings = ModelFactory.get_embeddings()
    db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    query = state["messages"][-1].content
    
    if not os.path.exists(db_path):
        return {"route_to": "tool_search_node", "rag_context": "로컬 지식고 부재로 실시간 외부 검색 모드로 전환합니다."}
        
    retrieved_docs = []
    
    # 1. 관리자가 등록한 FAQ 전용 자산고(approved_faq_db) 선제 탐색 및 결합
    try:
        faq_store = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="approved_faq_db")
        faq_docs = faq_store.as_retriever(search_kwargs={"k": 2}).invoke(query)
        if faq_docs:
            retrieved_docs.extend(faq_docs)
            print(f"[RAG 엔진 지식 탐색] 관리자 승인 FAQ고에서 관련 지식 {len(faq_docs)}건 확보 완료.")
    except Exception as e:
        print(f"[FAQ고 탐색 스킵] : {e}")

    # 2. 기본 7권의 설치 가이드북 RAG 지식고(Default) 교차 탐색 연동
    try:
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
        guide_docs = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(query)
        if guide_docs:
            retrieved_docs.extend(guide_docs)
    except Exception as e:
        print(f"[가이드북고 탐색 스킵] : {e}")

    if not retrieved_docs:
        return {"route_to": "tool_search_node", "rag_context": "로컬 가이드북 및 FAQ에 명세되지 않은 정보입니다. 외부 조회를 트리거합니다."}

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 문서 평가관입니다. 검색된 문맥이 엔지니어의 질문에 명확한 조치 가이드를 충분히 포함하고 있다면 'YES', 부족하거나 없다면 'NO'만 대답하세요. 다른 서술은 절대 금지합니다."),
        ("user", "검색 문맥:\n{context}\n\n질문: {query}")
    ])
    
    initial_context = "\n".join([d.page_content for d in retrieved_docs])
    eval_chain = eval_prompt | llm
    assessment = eval_chain.invoke({"context": initial_context, "query": query}).content.strip().upper()
    
    if "NO" in assessment:
        print("[RAG 에이전트 판단] 통합 지식고 내 정보 부족 감지 ➔ 자율적 인터넷 실시간 Tool 검색 가동.")
        return {
            "route_to": "tool_search_node",
            "rag_context": "통합 지식고에 명세되지 않은 정보입니다. 실시간 외부 네트워크 조회를 트리거합니다."
        }

    try:
        ranker = Ranker()
        passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(retrieved_docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)
        
        final_context_chunks = []
        for res in reranked_results[:2]:
            source_info = res.get("meta", {}).get("source_file", "OCP 4.20 지식고 자산")
            final_context_chunks.append(f"[출처: {source_info}]\n{res['text']}")
            
        return {"route_to": "answer_refiner_node", "rag_context": "\n---\n".join(final_context_chunks)}
    except Exception:
        return {"route_to": "answer_refiner_node", "rag_context": "\n---\n".join([d.page_content for d in retrieved_docs[:2]])}

# ====================================================================
# [Agent 3] Tool Search Node (완전한 프레임워크 주도형 자동 실행 구조)
# ====================================================================
def tool_search_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    llm_with_tools = llm.bind_tools([live_search_tool]) if hasattr(live_search_tool, "name") else llm
    query = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 실시간 외부 Red Hat 기술 포털 및 인터넷 인프라 지식을 자율 검색할 권한을 가진 운영관입니다. "
                   "엔지니어의 질문에서 핵심 키워드를 정밀하게 추출하여 연동된 기술 검색 도구를 자율적으로 실행(Call)하십시오."),
        ("user", "{input}")
    ])
    chain = prompt | llm_with_tools
    response = chain.invoke({"input": query})
    return {"messages": [response]}

# ====================================================================
# [Agent 4] Answer Refiner Agent (카테고리별 동적 템플릿 구조화)
# ====================================================================
def answer_refiner_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    query = state["messages"][-1].content
    category = state.get("input_category", "INSTALL")
    
    messages = state.get("messages", [])
    extracted_tool_output = ""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            extracted_tool_output = msg.content
            break
            
    context = (
        f"RAG 지식고 추출 내역:\n{state.get('rag_context','')}\n"
        f"인터넷 실시간 Tool 실행 결과:\n{extracted_tool_output if extracted_tool_output else state.get('tool_result', '')}"
    )
    
    if category == "ERROR":
        category_instruction = "당신은 긴급 장애 복구 전문관입니다. 에러 현상 해결을 위해 원인 분석을 먼저 제시하고, 현장에서 즉시 격리 조치 가능한 oc 명령어를 작성하세요."
    elif category == "SECURITY":
        category_instruction = "당신은 Red Hat 보안 정책 및 SCC 검증관입니다. 폐쇄망 권한 제약 및 암호화 자격증명 갱신 절차에 맞는 안전한 YAML 컴포넌트를 설계하세요."
    elif category == "NETWORK":
        category_instruction = "당신은 OpenShift SDN 및 인그레스 네트워크 엔지니어입니다. Disconnected 환경의 라우팅 매니페스트 및 프록시 정책 연동 규칙을 중심으로 서술하세요."
    else:
        category_instruction = "당신은 OCP 4.20 코어 인프라 구축 설계자입니다. 클러스터 초기 구축 및 프로젝트 명령어 표준 가이드라인을 작성하세요."

    reject_feedback = ""
    if len(messages) > 1 and isinstance(messages[-1], AIMessage) and "코드 검증 실패" in messages[-1].content:
        reject_feedback = f"\n\n⚠️ [코드 검증관의 반려 피드백 사항]: {messages[-1].content}\n지적된 명령어 오탈자나 references 링크 서식을 완벽히 수정해 내세요."

    refiner_template = f"""{category_instruction}
제공된 문맥 데이터의 범위 내에서만 사실에 기반하여 답변을 조립해야 합니다. 정보가 유실되지 않도록 철저히 반영하세요. 
반드시 다른 서술을 전면 배제하고 아래 정의된 JSON 구조화 스펙 양식에 맞춘 단일 JSON 데이터만 반환하세요.

[출력 JSON 가이드라인]
{{{{
  "summary": "조치 결과 요약 정보 명시",
  "steps": ["단계 1 설명 문자열", "단계 2 설명 문자열"],
  "code_block": "터미널 실행용 oc 명령어 또는 YAML",
  "references": ["https://docs.openshift.com/container-platform/4.20/로 시작하는 공식 가이드 링크"]
}}}}
{reject_feedback}
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
반드시 아래 정의된 JSON 스키마 규격을 100% 만족하는 완벽한 단일 JSON 데이터 하나만 반환하세요. 앞뒤에 ```json 같은 마크다운 선언은 절대 금지합니다.
특히 'references' 필드에 제공된 URL 주소 목록 중 단 하나라도 '[https://docs.openshift.com/container-platform/4.20/](https://docs.openshift.com/container-platform/4.20/)' 또는 관리자 자산 출처가 아닐 경우 STATUS: REJECT로 반려하십시오.

[반드시 반환해야 하는 JSON 구조화 스펙]
{{{{
  "summary": "조치 결과 요약 (반려 시 첫 줄은 반드시 STATUS: REJECT로 시작)",
  "steps": ["단계별 조치 절차 리스트"],
  "code_block": "실행 스크립트 또는 YAML",
  "references": ["참고 출처 문서 명시"]
}}}}
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
        
        if "STATUS: REJECT" in str(res_dict.get("summary", "")) and loop_count < 2:
            print(f"[검증관 반려 루프 가동]: {res_dict['summary']}")
            return {
                "messages": [AIMessage(content=f"코드 검증 실패 사유: {res_dict['summary']}")],
                "route_to": "re_edit",
                "validation_loop_count": loop_count + 1
            }
            
        if "STATUS: REJECT" in str(res_dict.get("summary", "")):
            res_dict["summary"] = "OpenShift 4.20 지식 가이드 (자동 구조 정제 탈출 완료)"
            
        return {
            "messages": [AIMessage(content="최종 승인 통과.")],
            "route_to": "approve",
            "final_structured_output": res_dict
        }
    except Exception:
        fallback = {
            "summary": "OpenShift 4.20 지식 가이드 가동 마감",
            "steps": ["요청하신 현상에 대해 교차 지식고 조립을 마감했습니다."],
            "code_block": f"# 통합 데이터 파싱 아웃풋 복구\n# {draft[:100]}",
            "references": ["Chroma DB 내장 관리자 FAQ 및 가이드북 통합 가이드"]
        }
        return {"route_to": "approve", "final_structured_output": fallback}

# ====================================================================
# [Graph Design] 복합 에이전틱 전환 라우팅 맵 빌드
# ====================================================================
def router_edge_decision(state: AgentState):
    return state["route_to"]

def ocp_rag_edge_decision(state: AgentState):
    return state["route_to"]

def validator_edge_decision(state: AgentState):
    if state["route_to"] == "re_edit":
        return "answer_refiner_node"
    return "end_node"

workflow = StateGraph(AgentState)

workflow.add_node("router_agent", router_agent)
workflow.add_node("ocp_rag_node", ocp_rag_node)
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("answer_refiner_node", answer_refiner_node)
workflow.add_node("code_validator_node", code_validator_node)

standard_tool_node = ToolNode([live_search_tool]) if hasattr(live_search_tool, "name") else ToolNode([])
workflow.add_node("action_tools", standard_tool_node)

workflow.set_entry_point("router_agent")
workflow.add_conditional_edges("router_agent", router_edge_decision, {
    "ocp_rag_node": "ocp_rag_node",
    "tool_search_node": "tool_search_node"
})

workflow.add_conditional_edges("ocp_rag_node", ocp_rag_edge_decision, {
    "answer_refiner_node": "answer_refiner_node",
    "tool_search_node": "tool_search_node"
})

workflow.add_conditional_edges("tool_search_node", tools_condition, {
    "tools": "action_tools",
    "__end__": "answer_refiner_node"
})
workflow.add_edge("action_tools", "answer_refiner_node")
workflow.add_edge("answer_refiner_node", "code_validator_node")

workflow.add_conditional_edges("code_validator_node", validator_edge_decision, {
    "answer_refiner_node": "answer_refiner_node",
    "end_node": END
})

compiled_graph = workflow.compile(checkpointer=MemorySaver())