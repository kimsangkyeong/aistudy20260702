import os
from typing import List, Dict, Any, TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain 및 LangGraph 컴포넌트 로드
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from graphhooks import add_messages # State 메시지 합산용 오퍼레이터 대체용 기본 스키마 정의
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 앞서 설계한 모델 팩토리에서 동적 모델 로드 (가정)
from model_factory import ModelFactory

load_dotenv()

# ==========================================
# 1. Structured Output을 위한 응답 스키마 정의
# ==========================================
class OCPResponseSchema(BaseModel):
    """OCP-Ops 에이전트의 최종 구조화된 응답 포맷"""
    summary: str = Field(description="처리 결과에 대한 한 줄 요약")
    steps: List[str] = Field(description="폐쇄망 환경에서 수행해야 하는 단계별 가이드 및 조치 절차")
    code_block: str = Field(description="즉시 복사하여 사용할 수 있는 'oc' 명령어 또는 YAML 매니페스트")
    references: List[str] = Field(description="참고한 Red Hat 공식 문서 출처 및 링크")

# ==========================================
# 2. LangGraph 대화 상태(State) 정의
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # 대화 이력 관리
    route_to: str                                        # 라우터가 결정한 다음 노드 방향
    rag_context: str                                     # RAG를 통해 검색된 사내 지식 컨텍스트
    tool_result: str                                     # 외부 Tool 호출을 통해 확보한 최신 트렌드 지식

# ==========================================
# 3. 각 Multi-Agent 노드(Node) 및 기능 구현
# ==========================================

# [Node 1] Router Agent: 사용자 의도 분석 및 분류
def router_agent(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    last_message = state["messages"][-1].content
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 들어온 요청을 라우팅하는 전문 라우터 에이전트입니다. "
                   "사용자의 질문이 OCP 4.20 설치/기본 명령어나 고정 FAQ 범위 내라면 'rag'를 반환하고, "
                   "최신 버그/에러/실시간 이슈 추적이 필요하다면 'tool'을 반환하세요. "
                   "오직 'rag' 또는 'tool' 한 단어만 반환해야 합니다."),
        ("user", "{input}")
    ])
    
    chain = router_prompt | llm
    response = chain.invoke({"input": last_message})
    route_decision = response.content.strip().lower()
    
    # 예외 방지용 방어 코드
    if "tool" in route_decision:
        return {"route_to": "tool_search_node"}
    return {"route_to": "ocp_rag_node"}


# [Node 2] OCP RAG Agent: Chroma DB 기반 내부 공식 지식 검색
def ocp_rag_node(state: AgentState) -> Dict[str, Any]:
    # 임베딩 팩토리와 연동하여 로컬 Chroma Vector DB 로드
    embeddings = ModelFactory.get_embeddings()
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # 1주 완성 가이드: Chroma 로드 및 가상 검색 구현 (실제 DB 구성 시 교체)
    # from langchain_community.vectorstores import Chroma
    # vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    # docs = vector_store.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(state["messages"][-1].content)
    # context = "\n".join([d.page_content for d in docs])
    
    print("[Node] OCP_RAG_Agent 작동: Chroma DB 검색 수행 완료.")
    mock_context = "Red Hat OpenShift 4.20 가이드: 폐쇄망 환경에서는 자격 증명 업데이트 시 'oc set data secret/pull-secret -n openshift-config' 명령어를 사용하여 로컬 레지스트리 토큰을 갱신해야 합니다."
    return {"rag_context": mock_context}


# [Node 3] Tool Search Node: 외부 Red Hat 포털 실시간 크롤링 및 API 조회 (Tavily 대체)
def tool_search_node(state: AgentState) -> Dict[str, Any]:
    print("[Node] Tool_Search_Agent 작동: Red Hat Errata 및 실시간 외부 검색 API 가동.")
    # 실제 환경에서는 TavilySearchResults 이나 커스텀 Red Hat API Request 툴 연동
    fetched_tool_result = "[최신 이슈 툴팁] OCP 4.20.1 일부 빌드에서 pull-secret 미러링 시 인코딩 오류가 보고됨. --from-file 플래그 뒤에 올바른 JSON 경로 명시 필수."
    return {"tool_result": fetched_tool_result}


# [Node 4] Answer Refiner Agent + Code Validator: 교차 검증 및 Structured Output 생성
def answer_refiner_node(state: AgentState) -> Dict[str, Any]:
    llm = ModelFactory.get_llm()
    user_query = state["messages"][-1].content
    context = state.get("rag_context", "") + "\n" + state.get("tool_result", "")
    
    refiner_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 Red Hat OCP 4.20 인프라 전문가이자 코드 검증관입니다. "
                   "주어진 문맥(Context)만을 철저히 바탕으로, 폐쇄망에서 즉시 실행 가능한 정확한 명령어 가이드를 작성하세요. "
                   "문맥에 없는 가상 내용을 지어내어 환각을 일으키면 감점됩니다. CoT(Chain of Thought) 방식으로 내부 검증을 거치세요.\n\n"
                   "참고할 문맥:\n{context}"),
        ("user", "{query}")
    ])
    
    # LangChain 고급 기능: Structured Output 강제 연동 (.with_structured_output)
    # Azure OpenAI와 Gemini 모두 최신 LangChain 버전에선 이 인터페이스를 공통 지원합니다.
    structured_llm = llm.with_structured_output(OCPResponseSchema)
    chain = refiner_prompt | structured_llm
    
    final_output = chain.invoke({"context": context, "query": user_query})
    
    # UI에 전달하기 쉽게 AIMessage 내부에 파싱된 데이터 저장 형태로 리턴
    return {"messages": [AIMessage(content=str(final_output.dict()))]}

# ==========================================
# 4. LangGraph 워크플로우 정의 및 컴파일
# ==========================================
def conditional_route(state: AgentState):
    """라우터 에이전트의 판단에 따라 분기 처리하는 엣지 로직"""
    return state["route_to"]

workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("router_agent", router_agent)
workflow.add_node("ocp_rag_node", ocp_rag_node)
workflow.add_node("tool_search_node", tool_search_node)
workflow.add_node("answer_refiner_node", answer_refiner_node)

# 흐름 연결 (Edge 설정)
workflow.set_entry_point("router_agent")

# 조건부 분기 연결 (Router -> RAG 또는 Tool)
workflow.add_conditional_edges(
    "router_agent",
    conditional_route,
    {
        "ocp_rag_node": "ocp_rag_node",
        "tool_search_node": "tool_search_node"
    }
)

# 무조건 최종 조립 노드로 연결
workflow.add_edge("ocp_rag_node", "answer_refiner_node")
workflow.add_edge("tool_search_node", "answer_refiner_node")
workflow.add_edge("answer_refiner_node", END)

# 메모리 저장을 위한 체크포인터 등록 (과제 필수 요건 반영)
memory = MemorySaver()
compiled_agent = workflow.compile(checkpointer=memory)