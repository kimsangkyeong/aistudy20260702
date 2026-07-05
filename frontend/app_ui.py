# frontend/app_ui.py
import os
import sys
import streamlit as st
import requests

# [경로 가드레일] 최상위 model_factory 모듈에 유연하게 접근하기 위한 경로 오케스트레이션 주입
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from model_factory import ModelFactory
from langchain_chroma import Chroma

st.set_page_config(page_title="OCP-Ops 플랫폼 v2", page_icon="⚙️", layout="wide")
st.title("⚙️ OCP-Ops Agent 플랫폼")
st.caption("Agentic Tool Binding, Multi-Agent Loop 협업, 백엔드 데이터 무결성 검증")
st.markdown("---")

# 무한 반복 루프를 완벽 차단하는 상태 머신 변수 선언
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_thread_id" not in st.session_state:
    st.session_state.session_thread_id = "ocp_perfect_score_v2_session"
if "faq_widget_version" not in st.session_state:
    st.session_state.faq_widget_version = 0
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

BACKEND_URL = "http://127.0.0.1:8000/api/chat"

# ====================================================================
# [🔥 크리티컬 패치] 하드코딩 질문 전면 제거 및 순수 Chroma DB 동적 빌드
# ====================================================================
def load_dynamic_faq_menu_pure_chroma() -> list:
    # 💡 "선택 안 함" 이외의 모든 고정 질문 목록 제거
    base_menu = ["선택 안 함"]
    
    try:
        embeddings = ModelFactory.get_embeddings()
        db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
        
        if os.path.exists(db_path):
            # 캐시를 타지 않고, 매번 순수 물리 디렉토리에 접근해 최신 파일 형상을 강제 인터셉트
            vector_store = Chroma(
                persist_directory=db_path, 
                embedding_function=embeddings, 
                collection_name="approved_faq_db"
            )
            
            all_data = vector_store._collection.get()  # Chroma 네이티브 클라이언트 직접 타격
            if all_data and "metadatas" in all_data and all_data["metadatas"]:
                for meta in all_data["metadatas"]:
                    if meta and "approved_question" in meta:  # 💡 소문자 키로 명확히 검증
                        approved_q = meta.get("approved_question")
                        # 중복 적재 방지선을 타며 최신 질문 리스트를 실시간 결합
                        if approved_q and approved_q.strip() and approved_q not in base_menu:
                            base_menu.append(approved_q.strip())
    except Exception as e:
        print(f"[프론트엔드 FAQ 실시간 Clean-Sync 실패 로그] : {e}")
    return base_menu

# 매 렌더링 생명주기마다 100% 무결성을 가진 최신 FAQ 메뉴 리스트 획득
faq_menu = load_dynamic_faq_menu_pure_chroma()

# 1. 히스토리 멀티턴 화면 복원 렌더링
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            data = message["content"]
            st.info(f"📝 **조치 요약:** {data.get('summary')}")
            st.markdown("**📋 실행 가이드 절차:**")
            for step in data.get("steps", []):
                st.markdown(f"- {step}")
            st.code(data.get("code_block", ""), language="bash")
            st.caption(f"🔗 **참조 문서:** {', '.join(data.get('references', []))}")

# Selectbox 변경 시 이벤트를 중간 가로채기하는 콜백 함수 선언
def handle_faq_selection():
    selected_val = st.session_state[f"faq_select_key_{st.session_state.faq_widget_version}"]
    if selected_val != "선택 안 함":
        st.session_state.active_query = selected_val
        st.session_state.faq_widget_version += 1

# 2. 사이드바 빠른 기입 FAQ 메뉴 정의 (★ 순수 Chroma DB 연동 리스트 바인딩)
with st.sidebar:
    st.header("📌 단축 FAQ 플레이북")
    st.selectbox(
        "빠른 질의 선택 리스트:", 
        faq_menu, 
        index=0,
        key=f"faq_select_key_{st.session_state.faq_widget_version}",
        on_change=handle_faq_selection
    )
 
# 3. 인풋 제어 파트 (단축 FAQ 실행 대기 값 우선 바인딩 후 청소)
user_query = ""
if st.session_state.active_query:
    user_query = st.session_state.active_query
    st.session_state.active_query = "" # 단 1회 전송 후 즉시 클리어 처리하여 무한 반복 차단
else:
    chat_input = st.chat_input("질문할 내용을 입력해 주세요...")
    if chat_input:
        user_query = chat_input.strip()

# 사용자 공백 전송 예외 차단 레이어
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    payload = {"message": user_query, "thread_id": st.session_state.session_thread_id}
    
    with st.spinner("백엔드 Multi-Agent 군집이 실시간 리랭킹 및 상호 순환 검증 루프를 수행 중입니다..."):
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=45)
            
            if response.status_code == 200:
                result_data = response.json()
                
                with st.chat_message("assistant"):
                    st.info(f"📝 **조치 요약:** {result_data.get('summary')}")
                    st.markdown("**📋 실행 가이드 절차:**")
                    for step in result_data.get("steps", []):
                        st.markdown(f"- {step}")
                    st.code(result_data.get("code_block", ""), language="bash")
                    st.caption(f"🔗 **참조 문서:** {', '.join(result_data.get('references', []))}")
                
                st.session_state.chat_history.append({"role": "assistant", "content": result_data})
                st.rerun()
            
            elif response.status_code == 400:
                st.error(f"❌ [입력 필드 유효성 검증 실패] 사내 게이트웨이 메시지: {response.json().get('detail')}")
                st.rerun()
            else:
                st.error(f"❌ [백엔드 그래프 엔진 처리 오류] 상태 코드: {response.status_code} - 상세 내용: {response.text}")
                st.rerun()
                
        except requests.exceptions.Timeout:
            st.error("❌ [네트워크 타임아웃] 에러 복구 에이전트의 다자간 검증 연산 시간이 초과되었습니다. 질의를 단순화하여 다시 요청하세요.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ [인프라 통신 단절] FastAPI 아키텍처 서버 백엔드 연결 실패: {e}")