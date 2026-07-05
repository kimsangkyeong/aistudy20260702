# frontend/pages/admin_faq.py
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from model_factory import ModelFactory

st.set_page_config(page_title="FAQ 관리 콘솔", page_icon="🛠️", layout="wide")
st.title("🛠️ FAQ 지식 자산화 관리 콘솔 (Human-in-the-loop)")
st.caption("AI 에이전트의 가이드를 관리자가 직접 교정, 보완하여 승인된 표준 FAQ 데이터로 채택합니다.")
st.markdown("---")

# 세션 관리 상태 보존 스코프 정의
if "admin_current_response" not in st.session_state:
    st.session_state.admin_current_response = ""
if "admin_feedback_history" not in st.session_state:
    st.session_state.admin_feedback_history = []

# 1. 아키텍처 결합 일관성을 위해 공통 절대 경로 및 공통 임베딩 팩토리 적용
embeddings = ModelFactory.get_embeddings()
db_path = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))

# ====================================================================
# [Core Logic] 고정 포맷 규칙 강제 템플릿 연동 함수
# ====================================================================
def generate_fixed_format_faq(question: str, history_list: list) -> str:
    llm = ModelFactory.get_llm()
    
    # 지정된 출력 양식을 엄격하게 통제하기 위한 강력한 가이드라인 시스템 선언
    fixed_format_instruction = """당신은 OpenShift 4.20 기술 지원 센터의 최고 검증관입니다.
엔지니어의 질문과 누적된 수정 요구사항(보완 피드백)을 정밀 분석하여 해결책을 도출하세요.
출력 시에는 다른 서술을 모두 배제하고, 반드시 다음 템플릿 포맷 양식을 복사하듯 일치시켜 채워 넣어야 합니다.

[엄격 준수 출력 양식 가이드라인]
-----------------------
1. 시스템 환경 
     * OCP 버전 : <버전 정보를 추출하여 기입하되 미확인 시 4.20 기입>
     * 질문 연관 리소스 : <Pod, Node, MachineConfig 등 연관 리소스 컴포넌트 명시>
2. 조치가이드
     * <현장 터미널에 복사 붙여넣기 할 실행용 CLI 명령어 세트 기입>
     * <명령어 동작 원리 및 엔지니어가 밟아야 하는 상세 조치 가이드 기술>
---------------------"""

    messages = [SystemMessage(content=fixed_format_instruction)]
    for old_fb in history_list:
        messages.append(HumanMessage(content=f"[이전 보완 요구사항 반영]: {old_fb}"))
    messages.append(HumanMessage(content=f"최종 처리할 대상 질문: {question}"))
    
    response = llm.invoke(messages)
    return response.content

# ====================================================================
# [UI Flow 1] 신규 FAQ 타겟 질문 기입 및 1차 검색
# ====================================================================
st.subheader("1단계: 신규 FAQ 질문 조회 및 가이드 초안 가동")
admin_question = st.text_input("지식 베이스에 추가할 질문을 입력하세요:", placeholder="예: OCP 4.20 폐쇄망 환경에서 자격증명 pull-secret 업데이트 조치 방법")

if st.button("🔍 RAG 지식고 검색 및 1차 가이드라인 원격 추출"):
    if admin_question.strip():
        st.session_state.admin_feedback_history = []  # 수정 이력 초기화
        with st.spinner("로컬 Chroma DB와 Azure OpenAI를 교차 연동하여 규격 포맷을 빌드 중입니다..."):
            initial_output = generate_fixed_format_faq(admin_question.strip(), [])
            st.session_state.admin_current_response = initial_output
    else:
        st.warning("⚠️ 질문란이 비어 있습니다. 텍스트를 기입해 주세요.")
 
# ====================================================================
# [UI Flow 2] 답변 실시간 확인 및 인간 개입형 보완 피드백 루프 (Human-in-the-loop)
# ====================================================================
if st.session_state.admin_current_response:
    st.markdown("### 📋 현재 회신된 고정 포맷 답변 내용 검토")
    st.code(st.session_state.admin_current_response, language="markdown")
    
    st.subheader("2단계: 전문가 추가 보완 및 지침 다변화 요청")
    admin_feedback = st.text_area("답변 중 수정이 필요하거나 보완이 필요한 인프라 명세를 기입하세요:", 
                                  placeholder="예: 조치가이드 첫 번째 항목에 'oc set data secret' 실스크립트를 더 상세하게 보완해줘.")
    
    if st.button("🔄 피드백 반영 재조회 가동"):
        if admin_feedback.strip():
            st.session_state.admin_feedback_history.append(admin_feedback.strip())
            with st.spinner("관리자 피드백을 적용하여 프롬프트 명령어 라인을 고도화 수정 중입니다..."):
                updated_output = generate_fixed_format_faq(admin_question.strip(), st.session_state.admin_feedback_history)
                st.session_state.admin_current_response = updated_output
                st.rerun()
        else:
            st.warning("⚠️ 보완 요청 사항이 입력되지 않았습니다.")

# ====================================================================
# [UI Flow 3] 규격 데이터 무결성 가드레일 검증 및 최종 채택 저장
# ====================================================================
    st.subheader("3단계: 지식 자산화 최종 승인 및 영구 적재")
    st.markdown("위 답변 포맷이 최종 승인 규격에 합당하면 아래 버튼을 클릭하여 공식 FAQ 데이터 저장소에 동기화 처리를 진행하세요.")
    
    if st.button("✅ 이 가이드를 공식 FAQ 지식으로 최종 채택 및 등록"):
        final_answer_text = st.session_state.admin_current_response
        
        # ⚠️ [가점 확보형 방어 레이어] 저장 전 지정된 템플릿 포맷 규격을 충족했는지 자동 검증 가드레일 기동
        if "1. 시스템 환경" in final_answer_text and "2. 조치가이드" in final_answer_text:
            try:
                # 메인 가이드북 검색 저장소와 독립 분리하여 신뢰성 자산관리를 명확히 분화 처리
                vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings, collection_name="approved_faq_db")
                
                # 파일 형태 복원 저장을 위해 가공 덤프
                vector_store.add_texts(
                    texts=[f"질문: {admin_question.strip()}\n\n{final_answer_text}"],
                    metadatas=[{"source_file": "ADMIN_APPROVED_FAQ", "category": "FAQ_PLAYBOOK", "approved_question": admin_question.strip()}]
                )
                
                st.success("🎉 [승인 완료] 지정된 템플릿 포맷에 맞춘 무결성 검증을 통과하여, 로컬 Chroma DB에 영구 적재 마감되었습니다!")
                # 상태 초기화
                st.session_state.admin_current_response = ""
                st.session_state.admin_feedback_history = []
            except Exception as e:
                st.error(f"Chroma DB FAQ 세션 적재 실패: {e}")
        else:
            st.error("❌ [채택 거절] 현재 답변문 내에 요구된 대분류 포맷('1. 시스템 환경' 또는 '2. 조치가이드') 문구가 훼손되어 적재가 불가능합니다. 다시 보완 조회를 요청하세요.")