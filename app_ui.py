import streamlit as st
import json
from langchain_core.messages import HumanMessage
from app_agent import compiled_agent

st.set_page_config(page_title="OCP-Ops 플랫폼", page_icon="⚙️", layout="wide")
st.title("⚙️ OCP-Ops Agent 플랫폼")
st.caption("OpenShift 4.20 Disconnected 환경 실무 운영 및 플레이북 질의 에이전트")
st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_thread_id" not in st.session_state:
    st.session_state.session_thread_id = "ocp_session_unique_101"

with st.sidebar:
    st.header("📌 단축 FAQ 가이드")
    faq_menu = [
        "선택 안 함",
        "OCP 4.20 폐쇄망 환경에서 pull-secret 자격증명 업데이트 방법",
        "Disconnected 환경 오프라인 레지스트리 미러링 설정 스크립트",
        "4.20 버전 보안 정책 변경에 따른 사내 NetworkPolicy 가이드"
    ]
    shortcut_selection = st.selectbox("빠른 질의 선택:", faq_menu)
    st.markdown("---")
    st.markdown("🌐 **현재 인프라 작동 모드:**")
    st.code(f"RUNNING_ENV 모드 활성화")

def render_json_ui(raw_string: str):
    try:
        clean_str = raw_string.replace("'", '"')
        parsed = json.loads(clean_str)
        
        st.subheader("📝 조치 사항 요약")
        st.info(parsed.get("summary", "데이터 없음"))
        
        st.subheader("📋 가이드 단계별 절차")
        for i, step in enumerate(parsed.get("steps", []), 1):
            st.markdown(f"**{i}.** {step}")
            
        st.subheader("💻 터미널 실행 코드 블록 (원클릭 카피)")
        st.code(parsed.get("code_block", "# 스크립트 없음"), language="bash")
        
        st.subheader("🔗 출처 근거 가이드 문서")
        for ref in parsed.get("references", []):
            st.caption(f"• {ref}")
    except Exception as e:
        st.warning(f"구조화 포맷 파싱 우회 모드로 출력합니다. (이유: {e})")
        st.write(raw_string)

final_prompt = ""
if shortcut_selection != "선택 안 함":
    final_prompt = shortcut_selection
else:
    user_type_in = st.chat_input("질문할 내용을 입력해 주세요...")
    if user_type_in:
        final_prompt = user_type_in

if final_prompt:
    with st.chat_message("user"):
        st.markdown(final_prompt)
    st.session_state.chat_history.append({"role": "user", "content": final_prompt})
    
    graph_config = {"configurable": {"thread_id": st.session_state.session_thread_id}}
    
    with st.spinner("Multi-Agent 아키텍처가 RAG 문서를 기반으로 교차 유효성 검증을 수행하고 있습니다..."):
        output_state = compiled_agent.invoke(
            {"messages": [HumanMessage(content=final_prompt)]}, 
            graph_config
        )
        ai_response_raw = output_state["messages"][-1].content
        
    with st.chat_message("assistant"):
        render_json_ui(ai_response_raw)
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response_raw})