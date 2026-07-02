import streamlit as st
import json
from langchain_core.messages import HumanMessage
from app_agent import compiled_agent  # 앞서 작성한 LangGraph 컴파일 객체 임포트

# ==========================================
# 1. 페이지 기본 설정 및 테마 정의
# ==========================================
st.set_page_config(
    page_title="OCP-Ops Agent 플랫폼",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 OCP-Ops Agent : OpenShift 4.20 폐쇄망 운영 가이드")
st.caption("Red Hat 공식 문서 RAG 및 실시간 검증 기반의 엔지니어 전용 멀티 에이전트 서비스")
st.markdown("---")

# Session State를 이용한 채팅 이력 및 데이터 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "ocp_heavy_user_001"  # LangGraph Memory 식별용 ID

# ==========================================
# 2. 사이드바 구성 (자주 묻는 질문 FAQ 단축 메뉴)
# ==========================================
with st.sidebar:
    st.header("📌 자주 묻는 FAQ 플레이북")
    st.subheader("원클릭으로 즉시 가이드를 확인하세요.")
    
    faq_list = [
        "선택 안 함",
        "OCP 4.20 폐쇄망 환경에서 pull-secret 자격증명 업데이트 방법",
        "Disconnected 환경 오프라인 레지스트리 미러링 설정 스크립트",
        "4.20 버전 보안 정책 변경에 따른 사내 NetworkPolicy 가이드",
        "폐쇄망 내부 오퍼레이터(Operator) 서브스크립션 카탈로그 소스 생성"
    ]
    
    selected_faq = st.selectbox("FAQ 목록을 탐색하세요:", faq_list)
    
    st.markdown("---")
    st.info(
        "💡 **운영 팁**\n\n"
        "회사(Azure GPT-4o)와 집(Gemini Pro) 가동 환경에 따라 "
        "자동으로 LLM과 임베딩 모델이 교체되므로 소스 코드를 별도 수정하지 않아도 됩니다."
    )

# ==========================================
# 3. 데이터 추론 및 UI 렌더링 헬퍼 함수
# ==========================================
def parse_and_show_response(response_content: str):
    """Structured Output(JSON 문자열)을 파싱하여 UI 컴포넌트로 배치"""
    try:
        # 문자열로 들어온 JSON 파싱
        data = json.loads(response_content.replace("'", '"')) # 싱글쿼테이션 방어 예외 처리
        
        # 1. 요약 정보
        st.subheader("📝 조치 요약")
        st.success(data.get("summary", "요약 정보를 불러올 수 없습니다."))
        
        #