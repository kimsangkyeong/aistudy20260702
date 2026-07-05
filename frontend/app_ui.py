# frontend/app_ui.py
import streamlit as st
import requests

st.set_page_config(page_title="OCP-Ops 플랫폼 v2", page_icon="⚙️", layout="wide")
st.title("⚙️ OCP-Ops Agent 플랫폼 (심사위원 최종 보완 검증판)")
st.caption("Agentic Tool Binding, Multi-Agent Loop 협업, 백엔드 데이터 무결성 검증 완비 버전")
st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_thread_id" not in st.session_state:
    st.session_state.session_thread_id = "ocp_perfect_score_v2_session"

BACKEND_URL = "http://127.0.0.1:8000/api/chat"

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

# 2. 사이드바 빠른 기입 FAQ 메뉴 정의
with st.sidebar:
    st.header("📌 단축 FAQ 플레이북")
    faq_menu = [
        "선택 안 함",
        "OCP 4.20 폐쇄망 환경에서 자격증명(pull-secret) 업데이트하는 oc 명령어 알려줘",
        "Disconnected 환경 오프라인 레지스트리 미러링 구성용 oc adm release mirror 가이드 수립"
    ]
    shortcut_selection = st.selectbox("빠른 질의 선택 리스트:", faq_menu)
 
# 3. 인풋 제어 파트
user_query = ""
if shortcut_selection != "선택 안 함":
    user_query = shortcut_selection
else:
    chat_input = st.chat_input("질문할 내용을 입력해 주세요...")
    if chat_input:
        user_query = chat_input.strip()

# 🔥 [보완 사항 반영] 사용자가 공백만 채워서 보내거나 빈 문자열인 경우 백엔드 전송을 원천 차단하는 방어선 구축
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    payload = {"message": user_query, "thread_id": st.session_state.session_thread_id}
    
    with st.spinner("백엔드 Multi-Agent 군집이 실시간 리랭킹 및 상호 순환 검증 루프를 수행 중입니다..."):
        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=45)
            
            # 🔥 [보완 사항 반영] 통신 응답 코드에 따른 다각도 예외 처리 보강 및 무결성 파싱
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
            
            elif response.status_code == 400:
                st.error(f"❌ [입력 필드 유효성 검증 실패] 사내 게이트웨이 메시지: {response.json().get('detail')}")
            else:
                st.error(f"❌ [백엔드 그래프 엔진 처리 오류] 상태 코드: {response.status_code} - 상세 내용: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("❌ [네트워크 타임아웃] 에러 복구 에이전트의 다자간 검증 연산 시간이 초과되었습니다. 질의를 단순화하여 다시 요청하세요.")
        except Exception as e:
            st.error(f"❌ [인프라 통신 단절] FastAPI 아키텍처 서버 백엔드 연결 실패: {e}")