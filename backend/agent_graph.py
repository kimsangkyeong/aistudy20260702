import os
import re
import json
import uuid
from typing import Dict, Any, TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# 🔴 [시나리오 B 적용] 종속성 충돌을 일으키던 Sqlite 관련 인터페이스를 완전히 도려내고 MemorySaver 확정 동기화
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

# 실제 연동할 Red Hat 공식 기술 포털 실시간 검색 도구 객체 확보
live_search_tool = get_redhat_live_search_tool()

# ====================================================================
# [Agent 1] Router Agent (의도 및 기술 카테고리 분류 관문 에이전트)
# ====================================================================
def router_agent(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    last_query = state["messages"][-1].content
    
    router_few_shot = """당신은 인프라 엔지니어의 질문 의도와 기술 범주를 정확히 분류하는 지능형 기술 라우터 에이전트입니다.
반드시 제공된 <query_context> 내부의 조건과 Few-shot 패턴을 비교 분석하여 결과를 '라우팅_방향|기술_카테고리' 포맷 규칙에 맞춰 출력하세요. 다른 잡설은 전면 차단합니다.

[정밀 구조화 Few-shot 예시]
<example>
  <query_context>OCP 4.20 폐쇄망 registry 미러링 설정을 위한 mirror-config.yaml 예시 보여줘</query_context>
  <output>rag|INSTALL</output>
</example>
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", router_few_shot),
        ("user", "<query_context>{input}</query_context>")
    ])
    chain = prompt | llm
    raw_decision = chain.invoke({"input": last_query}).content.strip()
    
    parsed_success = False
    if "|" in raw_decision:
        parts = raw_decision.split("|")
        if len(parts) >= 2:
            route = parts[0].strip().lower()
            category = parts[1].strip().upper()
            if route in ["rag", "tool"] and category in ["INSTALL", "NETWORK", "SECURITY", "ERROR"]:
                parsed_success = True
                
    if not parsed_success:
        text_clean = last_query.lower()
        route = "tool" if any(k in text_clean for k in ["에러", "error", "장애", "멈춤", "fail", "로그"]) else "rag"
        if any(k in text_clean for k in ["scc", "보안", "인증", "secret"]): category = "SECURITY"
        elif any(k in text_clean for k in ["network", "네트워크", "프록시", "proxy", "ingress"]): category = "NETWORK"
        elif any(k in text_clean for k in ["에러", "error", "장애", "fail"]): category = "ERROR"
        else: category = "INSTALL"
        
    return {
        "route_to": "tool_search_node" if route == "tool" else "ocp_rag_node",
        "input_category": category
    }

# ====================================================================
# [Agent 2] OCP Self-RAG Agent (지식 품질 적합성을 자율 채점 및 제어하는 독립 에이전트)
# ====================================================================
def ocp_rag_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    embeddings = ModelFactory.get_embeddings()
    db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    query = state["messages"][-1].content
    
    if not os.path.exists(db_path):
        return {"route_to": "tool_search_node", "rag_context": "로컬 지식고 부재로 실시간 외부 검색 모드로 전환합니다."}
        
    # FAQ 오탐 진압 벡터 스코어 대조 레이어 가동 (임계치 0.65 조율)
    try:
        faq_store = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="approved_faq_db")
        faq_results = faq_store.similarity_search_with_score(query, k=1)
        
        is_faq_matched = False
        target_doc = None
        target_q = ""
        
        if faq_results and faq_results[0][1] <= 0.65:
            target_doc = faq_results[0][0]
            target_q = target_doc.metadata.get("approved_question", "")
            is_faq_matched = True
            
        if not is_faq_matched:
            collection_data = faq_store._collection.get()
            if collection_data and "metadatas" in collection_data and collection_data["metadatas"]:
                for idx, meta in enumerate(collection_data["metadatas"]):
                    saved_q = meta.get("approved_question", "").strip()
                    if saved_q and (query.strip() in saved_q or saved_q in query.strip() or len(set(query) & set(saved_q)) / max(1, len(set(query))) > 0.75):
                        raw_content = collection_data["documents"][idx]
                        target_q = saved_q
                        from langchain_core.documents import Document
                        target_doc = Document(page_content=raw_content, metadata=meta)
                        is_faq_matched = True
                        break

        if is_faq_matched and target_doc:
            raw_doc = target_doc.page_content
            faq_structured_fallback = {
                "summary": f"💡 [관리자 공인 FAQ 동기화 응답] 시스템 관리자가 사내 인프라 표준으로 검증 및 채택한 공식 플레이북 자산입니다.\n질문 원문: {target_q}",
                "steps": [line.strip() for line in raw_doc.split("\n") if line.strip() and not line.strip().startswith("1. 시스템 환경") and not line.strip().startswith("2. 조치가이드")],
                "code_block": raw_doc.strip(),
                "references": ["Platform Ops 내장 승인 FAQ 지식고 (ADMIN_APPROVED_FAQ)"]
            }
            return {
                "route_to": "approve",
                "rag_context": raw_doc,
                "draft_answer": json.dumps(faq_structured_fallback, ensure_ascii=False),
                "final_structured_output": faq_structured_fallback
            }
            
    except Exception as ex:
        print(f"⚠️ [FAQ 정밀 대조 스킵 예외]: {ex}")

    # 일반 RAG 파이프라인 진행
    retrieved_docs = []
    try:
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
        guide_docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(query)
        if guide_docs: retrieved_docs.extend(guide_docs)
    except Exception: pass

    if not retrieved_docs:
        return {"route_to": "tool_search_node", "rag_context": "통합 지식고 정보 부족."}

    # 🔴 [리뷰 피드백 반영] 단순 노드를 독자 의사결정 'OCP_Self_RAG_Agent'로 격상 채점 유도
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 문서 평가 에이전트입니다. 검색된 문맥이 엔지니어의 질문에 명확한 가이드를 포함하고 있다면 'YES', 부족하다면 'NO'만 대답하세요."),
        ("user", "검색 문맥:\n{context}\n\n질문: {query}")
    ])
    initial_context = "\n".join([d.page_content for d in retrieved_docs])
    eval_chain = eval_prompt | llm
    assessment = eval_chain.invoke({"context": initial_context, "query": query}).content.strip().upper()
    
    if "NO" in assessment:
        print("[OCP_Self_RAG_Agent 판단] 자체 지식고 품질 한량 판정 -> 외부 수집 Analyst 에이전트에게 라우팅 자율 변경")
        return {
            "route_to": "tool_search_node",
            "rag_context": "로컬 지식고 정보 부족 판정으로 외부 네트워크 조회를 트리거합니다."
        }

    # FlashRank 코어로 실시간 순위 보정(Reranking) 수행
    try:
        ranker = Ranker()
        passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(retrieved_docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)
        final_chunks = [f"[출처: {res.get('meta', {}).get('source_file', 'OCP 가이드')}]\n{res['text']}" for res in reranked_results[:2]]
        return {"route_to": "answer_refiner_node", "rag_context": "\n---\n".join(final_chunks), "final_structured_output": {}}
    except Exception:
        return {"route_to": "answer_refiner_node", "rag_context": "\n---\n".join([d.page_content for d in retrieved_docs[:2]]), "final_structured_output": {}}

# ====================================================================
# [Agent 3] RedHat Live Analyst Agent (실시간 외부 기술 검색 및 가치 분석 에이전트)
# ====================================================================
def tool_search_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    llm_with_tools = llm.bind_tools([live_search_tool]) if hasattr(live_search_tool, "name") else llm
    query = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 외부 Red Hat 공식 포털 기술 데이터 분석 에이전트입니다. 엔지니어의 현장 질문에서 핵심 키워드를 추출하여 연동된 실시간 기술 검색 도구를 자율적으로 실행(Call)하십시오."),
        ("user", "{input}")
    ])
    chain = prompt | llm_with_tools
    response = chain.invoke({"input": query})
    return {"messages": [response], "final_structured_output": {}}

# ====================================================================
# [Agent 4] Answer Refiner Agent (제1 코어: 실무 밀착형 풍부한 설명을 직조하는 시니어 설계자)
# ====================================================================
def answer_refiner_node(state: AgentState) -> Dict[str, Any]:
    if state.get("final_structured_output") and "approved_question" in str(state["final_structured_output"].get("summary", "")):
        return {"draft_answer": json.dumps(state["final_structured_output"], ensure_ascii=False)}

    llm = ModelFactory.get_llm()
    query = state["messages"][-1].content
    category = state.get("input_category", "INSTALL")
    messages = state.get("messages", [])
    
    extracted_tool_output = next((msg.content for msg in reversed(messages) if isinstance(msg, ToolMessage)), "")
    context = f"RAG 지식고 추출 내역:\n{state.get('rag_context','')}\n인터넷 Tool 결과:\n{extracted_tool_output}"
    
    if category == "ERROR":
        category_instruction = "당신은 긴급 장애 복구 전문관입니다. 에러 현상의 근본적 원인(Root Cause) 분석과 인프라 파급 영향도를 상세히 서술하고, 현장에서 즉시 격리 조치 가능한 oc 명령어를 정밀하게 작성하세요."
    elif category == "SECURITY":
        category_instruction = "당신은 Red Hat 보안 정책 및 SCC 검증관입니다. 폐쇄망 환경의 보안 규정 제약 사항과 RBAC/SCC 자격증명 갱신 절차에 맞는 안전하고 풍부한 매니페스트 및 가이드를 서술하세요."
    elif category == "NETWORK":
        category_instruction = "당신은 OpenShift SDN 엔지니어입니다. 디스커넥티드 프록시 정책 연동, 인그레스 오브젝트 라우팅 규칙 및 트래픽 흐름을 중심으로 깊이 있게 서술하세요."
    else:
        category_instruction = "당신은 OCP 4.20 코어 인프라 구축 설계자입니다. 클러스터 초기 구축 시 유의사항과 아키텍처 토대를 포함하여 실무 명령어 표준 가이드라인을 상세히 작성하세요."

    reject_feedback = ""
    if len(messages) > 1 and isinstance(messages[-1], AIMessage) and "코드 검증 실패" in messages[-1].content:
        reject_feedback = f"\n\n⚠️ [독립 코드 검증관 에이전트의 REJECT 반려 지적사항]:\n{messages[-1].content}\n위 피드백 사유를 반영하여 명령어 파라미터 유효성 및 JSON 스펙을 철저히 보강 및 재작성하십시오."

    refiner_template = f"""{category_instruction}
제공된 참조 데이터를 바탕으로 현장 실무자에게 실질적인 기술적 분석을 제공할 수 있도록 친절하고 깊이 있게 설명하세요. 요약문이나 단답형 서술은 전면 금지합니다.
반드시 아래 정의된 JSON 구조화 스펙 양식에 100% 일치하게 데이터를 구성하세요.

[출력 JSON 구조 양식]
{{{{
  "summary": "단답형 요약을 금지하며, 현상의 기술적 원인 분석 및 실무 영향도를 최소 3문장 이상의 풍부한 인프라 관점으로 상세히 서술하세요.",
  "steps": [
    "1단계: 사전 환경 검증 및 대상 리소스 조회 (명령어 목적 포함 상세 서술)",
    "2단계: 실제 조치/변경 매니페스트 반영 및 실행 가이드 (옵션값 분석 포함)",
    "3단계: 사후 상태 검증 및 장애 재발 방지를 위한 영구 예방 팁 상세 설명"
  ],
  "code_block": "터미널 실행용 oc 명령어 세트 또는 완결성 있는 YAML 전체 매니페스트",
  "references": ["https://docs.openshift.com/container-platform/4.20/로 시작하는 공식 가이드 링크 또는 참조 아웃풋 파일명"]
}}}}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", refiner_template),
        ("user", "데이터:\n{context}\n\n질문: {query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return {"draft_answer": response.content}

# ====================================================================
# [Agent 5] Code Validator Agent (제2 코어: 독립 코드 검증관 에이전트)
# ====================================================================
def code_validator_node(state: AgentState) -> Dict[str, Any]:
    if state.get("final_structured_output") and "approved_question" in str(state["final_structured_output"].get("summary", "")):
        return {"route_to": "approve", "final_structured_output": state["final_structured_output"]}

    llm = ModelFactory.get_llm()
    draft = state.get("draft_answer", "")
    loop_count = state.get("validation_loop_count", 0)
    
    validator_template = """당신은 시니어 설계자 에이전트가 도출한 인프라 스크립트와 YAML 문법의 신뢰성을 완벽히 교차 검증하는 '독립 코드 검증관 에이전트'입니다.
정의된 구조화 스펙 및 명령어 파라미터 무결성을 엄격하게 판단하여, 조건 미달 시 반드시 첫 줄을 'STATUS: REJECT'로 시작하는 사유를 명시하여 반려하십시오.

[검증 및 반려(REJECT) 가이드라인]
1. 'references' 필드에 제공된 URL 주소 목록 중 단 하나라도 'https://docs.openshift.com/container-platform/4.20/' 또는 사내 자산 출처가 아닐 경우 STATUS: REJECT로 반려하십시오.
2. 'code_block' 내부에 단순 설명 주석만 존재하거나 oc 명령어가 완전히 누락된 경우 무조건 STATUS: REJECT로 반려하십시오.

    [반드시 반환해야 하는 JSON 구조화 스펙]
    {{{{
      "summary": "조치 결과 요약 (반려 시 첫 줄은 반드시 STATUS: REJECT로 시작)",
      "steps": ["단계별 절차"],
      "code_block": "실행 스크립트",
      "references": ["참고 출처 문서 명시"]
    }}}}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", validator_template),
        ("user", "가이드 초안:\n{draft_content}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(OCPResponseSchema)
        chain = prompt | structured_llm
        validated_obj = chain.invoke({"draft_content": draft})
        res_dict = validated_obj.dict()
        
        # 🔴 [리뷰 피드백 반영] 독립 에이전트의 강제 교차 재작성(re_edit) 튕겨내기 루프 제어
        if "STATUS: REJECT" in str(res_dict.get("summary", "")) and loop_count < 2:
            print(f"🔥 [독립 검증관 에이전트 분기 판단]: 초안 미흡으로 REJECT 반려 처리 (루프 카운트: {loop_count + 1}/2)")
            return {
                "messages": [AIMessage(content=f"코드 검증 실패 사유: {res_dict['summary']}")],
                "route_to": "re_edit",
                "validation_loop_count": loop_count + 1
            }
        return {"route_to": "approve", "final_structured_output": res_dict}
    except Exception:
        clean_draft = draft.replace("```json", "").replace("```", "").strip()
        try:
            parsed_json = json.loads(clean_draft)
            return {"route_to": "approve", "final_structured_output": parsed_json}
        except Exception:
            fallback = {
                "summary": "OpenShift 4.20 통합 장애 조치 가이드라인입니다.",
                "steps": ["사내 지식 베이스를 바탕으로 실무 조치 절차를 구성했습니다."],
                "code_block": draft,
                "references": ["Chroma DB 내장 가이드북 통합 분석 자산"]
            }
            return {"route_to": "approve", "final_structured_output": fallback}

# ====================================================================
# [Graph Design] 복합 에이전틱 전환 라우팅 맵 빌드 (LangGraph)
# ====================================================================
workflow = StateGraph(AgentState)
workflow.add_node("router_agent", router_agent)
workflow.add_node("ocp_rag_node", ocp_rag_node)
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("answer_refiner_node", answer_refiner_node)
workflow.add_node("code_validator_node", code_validator_node)

standard_tool_node = ToolNode([live_search_tool]) if hasattr(live_search_tool, "name") else ToolNode([])
workflow.add_node("action_tools", standard_tool_node)

workflow.set_entry_point("router_agent")
workflow.add_conditional_edges("router_agent", lambda x: x["route_to"], {"ocp_rag_node": "ocp_rag_node", "tool_search_node": "tool_search_node"})
workflow.add_conditional_edges("ocp_rag_node", lambda x: x["route_to"], {"answer_refiner_node": "answer_refiner_node", "tool_search_node": "tool_search_node", "approve": "code_validator_node"})
workflow.add_conditional_edges("tool_search_node", tools_condition, {"tools": "action_tools", "END": "answer_refiner_node"})
workflow.add_edge("action_tools", "answer_refiner_node")
workflow.add_edge("answer_refiner_node", "code_validator_node")

workflow.add_conditional_edges("code_validator_node", lambda x: "answer_refiner_node" if x["route_to"] == "re_edit" else "end_node", {
    "answer_refiner_node": "answer_refiner_node",
    "end_node": END
})

# 🔴 [시나리오 B 반영] 결함을 유발하던 SQLite 디스크 영속성 레이어를 전면 배제하고 MemorySaver 단독 컴파일 완수
compiled_graph = workflow.compile(checkpointer=MemorySaver())