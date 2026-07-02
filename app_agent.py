import os
from typing import List, Dict, Any, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from model_factory import ModelFactory

# 1. 아웃풋 데이터 규격 구조화 (Pydantic)
class OCPResponseSchema(BaseModel):
    summary: str = Field(description="조치 결과에 대한 엔지니어용 핵심 요약")
    steps: List[str] = Field(description="폐쇄망 타겟 노드에서 처리해야 할 단계별 가이드")
    code_block: str = Field(description="터미널에 즉시 실행 가능한 oc 명령어 세트 또는 YAML")
    references: List[str] = Field(description="출처 문서 장표 및 관련 Red Hat 레퍼런스")

# 2. 상태(State) 스키마 정의
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    route_to: str
    rag_context: str
    tool_result: str

# 3. Multi-Agent 핵심 노드 함수 구현
def router_agent(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    last_query = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 들어오는 질문의 특성을 분류하는 라우터 에이전트입니다. "
                   "질문이 OCP 4.20 매뉴얼 및 구축 절차 범위라면 'rag'를 반환하고, "
                   "실시간 버그 트래킹이나 긴급 에러 트러블슈팅 건이라면 'tool'을 반환하세요. "
                   "오직 'rag' 또는 'tool' 텍스트만 출력하세요."),
        ("user", "{input}")
    ])
    chain = prompt | llm
    decision = chain.invoke({"input": last_query}).content.strip().lower()
    return {"route_to": "tool_search_node" if "tool" in decision else "ocp_rag_node"}

def ocp_rag_node(state: AgentState) -> Dict[str, Any]:
    embeddings = ModelFactory.get_embeddings()
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    if not os.path.exists(db_path):
        return {"rag_context": "[경고] 생성된 RAG Vector DB가 없습니다. 먼저 ingest_pdf.py를 실행하세요."}
        
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    query = state["messages"][-1].content
    docs = vector_store.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
    context = "\n---\n".join([d.page_content for d in docs])
    return {"rag_context": context}

def tool_search_node(state: AgentState) -> Dict[str, Any]:
    # 외부 에러 트래킹을 모사한 Mock Tool 기능 가동
    return {"tool_result": "[Red Hat 포털 Errata 확인] OCP 4.20 일부 초기 빌드 호환 이슈 패치 정보 포함."}

def answer_refiner_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    query = state["messages"][-1].content
    context = f"RAG 지식:\n{state.get('rag_context','')}\n\n외부이슈 지식:\n{state.get('tool_result','')}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 Red Hat OCP 4.20 엔지니어이자 코드 유효성 검증관입니다. "
                   "제공된 문맥만을 바탕으로 사실에 근거한 실행 가이드를 CoT 형태로 완성하세요.\n\n"
                   "문맥 정보:\n{context}"),
        ("user", "{query}")
    ])
    
    structured_llm = llm.with_structured_output(OCPResponseSchema)
    chain = prompt | structured_llm
    final_json = chain.invoke({"context": context, "query": query})
    return {"messages": [AIMessage(content=str(final_json.dict()))]}

# 4. LangGraph 파이프라인 그래프 직조
def route_decision_edge(state: AgentState):
    return state["route_to"]

workflow = StateGraph(AgentState)
workflow.add_node("router_agent", router_agent)
workflow.add_node("ocp_rag_node", ocp_rag_node)
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("answer_refiner_node", answer_refiner_node)

workflow.set_entry_point("router_agent")
workflow.add_conditional_edges("router_agent", route_decision_edge, {
    "ocp_rag_node": "ocp_rag_node",
    "tool_search_node": "tool_search_node"
})
workflow.add_edge("ocp_rag_node", "answer_refiner_node")
workflow.add_edge("tool_search_node", "answer_refiner_node")
workflow.add_edge("answer_refiner_node", END)

compiled_agent = workflow.compile(checkpointer=MemorySaver())