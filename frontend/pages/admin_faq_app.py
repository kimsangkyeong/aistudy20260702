import os
import sys
import uuid
import streamlit as st
from dotenv import load_dotenv

# [경로 가드레일] 최상위 model_factory 모듈에 유연하게 접근하기 위한 경로 오케스트레이션
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from model_factory import ModelFactory
from langchain_chroma import Chroma

# 환경 변수 로딩
load_dotenv()

st.set_page_config(page_title="FAQ 관리자 콘솔 v3", page_icon="🔐", layout="wide")
st.title("🔐 OCP-Ops 플랫폼 : FAQ 지식 관리자 콘솔")
st.caption("Human-in-the-loop 기반 점진적 피드백 보완 루프 및 위젯 상태 무결성 동기화 레이어")
st.markdown("---")

# Form State 초기화 보완: 위젯 직접 수동 수정을 회피하기 위한 폼 전용 세션 상태 버퍼 가이드
if "form_question" not in st.session_state:
    st.session_state.form_question = ""
if "form_feedback" not in st.session_state:
    st.session_state.form_feedback = ""
if "admin_ai_draft" not in st.session_state:
    st.session_state.admin_ai_draft = ""
if "feedback_status_msg" not in st.session_state:
    st.session_state.feedback_status_msg = ""
if "feedback_error_msg" not in st.session_state:
    st.session_state.feedback_error_msg = ""

DB_PATH = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))

def get_faq_vector_store():
    embeddings = ModelFactory.get_embeddings()
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="approved_faq_db"
    )

# 화면 레이아웃 최적화를 위한 탭 분할
tab1, tab2 = st.tabs(["✨ 신규 FAQ 플레이북 자산 등록", "🗂️ 기존 적재 FAQ 자산 통합 조회 및 관리 (CRUD)"])

# ====================================================================
# [Tab 1] 신규 FAQ 플레이북 자산 등록 (점진적 피드백 보완 루프)
# ====================================================================
with tab1:
    st.header("✨ Human-in-the-loop 점진적 FAQ 빌드 파이프라인")
    
    # 1. 등록 및 검증할 신규 인프라 질문 입력 box (세션 버퍼 변수 바인딩으로 락 차단)
    admin_question = st.text_input("💡 등록 및 검증할 신규 인프라 질문 입력:", value=st.session_state.form_question, key="admin_question_reg_input")
    
    # 2. 질문 입력 box 밑에 "RAG 기반 AI 초안 가이드 생성" 버튼 배치
    if st.button("🚀 RAG 기반 AI 초안 가이드 생성", key="btn_generate_initial_draft"):
        if admin_question.strip():
            with st.spinner("사내 가이드북 컨텍스트를 스캔하여 지정 표준 포맷으로 초안을 조립 중..."):
                try:
                    llm = ModelFactory.get_llm()
                    
                    # 로컬 가이드북 기본 지식고 교차 탐색
                    guide_context = ""
                    try:
                        embeddings = ModelFactory.get_embeddings()
                        base_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
                        docs = base_store.as_retriever(search_kwargs={"k": 2}).invoke(admin_question)
                        guide_context = "\n".join([d.page_content for d in docs])
                    except Exception:
                        guide_context = "로컬 가이드북 기본 지식고 준비 대기 중"

                    initial_template = """당신은 OCP 4.20 코어 인프라 구축 설계자입니다. 제공된 문맥을 기반으로 사실에 입각하여 답변을 완성하세요.
반드시 다른 서론/결론을 전면 배제하고 아래 정의된 포맷 양식 규격을 100% 준수하여 출력하세요.

1. 시스템 환경
- OCP 버전 : 4.20
- 질문 연관 리소스 : 연관 컴포넌트 명시
2. 조치가이드
- 실행 명령어 및 단계별 조치 가이드 기술"""

                    from langchain_core.prompts import ChatPromptTemplate
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", initial_template),
                        ("user", "기본 문맥:\n{context}\n\n질문: {query}")
                    ])
                    chain = prompt | llm
                    response = chain.invoke({"context": guide_context, "query": admin_question})
                    
                    st.session_state.admin_ai_draft = response.content
                    st.session_state.form_question = admin_question  # 현재 타이핑 상태 보존
                    st.session_state.feedback_status_msg = ""
                    st.session_state.feedback_error_msg = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"초안 생성 중 런타임 예외 발생: {e}")
        else:
            st.error("질문 내용을 먼저 기입해 주세요.")
            
    # 3. 그 밑에 답변 창(편집 공간) 표시 -> 높이 2배 상향 (600)
    st.markdown("### 📋 최종 검증 및 편집 공간")
    final_answer_text = st.text_area(
        "🤖 AI 가이드라인 최종 텍스트 검증 및 편집 수정 (순수 본문 영역):", 
        value=st.session_state.admin_ai_draft, 
        height=600,
        key="final_answer_text_area"
    )
    
    st.markdown("---")
    
    # 레이아웃 요구사항: 추가 보완 요구사항 항목들과 FAQ 등록 버튼을 7:3 구조로 행 배치 분할
    col_feedback, col_reg_btn = st.columns([7, 3])
    
    with col_feedback:
        # 4. 답변 창 밑에 추가 보완 명세 요구사항 입력 box 위치
        admin_feedback = st.text_input("📝 답변 추가 보완 명세 요구사항 입력 (선택):", value=st.session_state.form_feedback, key="admin_feedback_reg_input")
        
        # 5. 그 밑에 "추가 보완 포함 가이드 생성" 버튼 생성
        if st.button("🔄 추가 보완 포함 가이드 생성", key="btn_generate_refined_guide"):
            if admin_question.strip() and final_answer_text.strip() and admin_feedback.strip():
                with st.spinner("이전 답변과 추가 보완 요구사항을 병합하여 가이드를 정밀 재구성 중..."):
                    try:
                        llm = ModelFactory.get_llm()
                        
                        refiner_template = f"""당신은 OCP 4.20 코어 인프라 구축 설계자입니다.
기존 조치 가이드 초안 내용하고 관리자의 추가 요구사항을 완벽히 병합하여, 유실 없는 최종 고도화 답변을 완성하세요.
반드시 다른 서론/결론을 전면 배제하고 아래 정의된 포맷 양식 규격을 100% 준수하여 출력하세요.

1. 시스템 환경
- OCP 버전 : 4.20
- 질문 연관 리소스 : 연관 컴포넌트 명시
2. 조치가이드
- 실행 명령어 및 단계별 조치 가이드 기술

⚠️ [관리자의 추가 보완 명세 요구사항]: {admin_feedback.strip()}"""

                        from langchain_core.prompts import ChatPromptTemplate
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", refiner_template),
                            ("user", "기본 가이드 초안:\n{previous_draft}\n\n질문 원문: {query}")
                        ])
                        chain = prompt | llm
                        response = chain.invoke({"previous_draft": final_answer_text, "query": admin_question})
                        
                        # 상태 홀딩 업데이트 및 피드백 메시지 설정
                        st.session_state.admin_ai_draft = response.content
                        st.session_state.form_question = admin_question
                        st.session_state.form_feedback = admin_feedback
                        st.session_state.feedback_status_msg = "📝 최신 보완본이 반영되었습니다. 위의 '최종 검증 및 편집 공간' 내용을 확인하세요!"
                        st.session_state.feedback_error_msg = ""
                        st.rerun()
                    except Exception as e:
                        st.session_state.feedback_error_msg = f"❌ 보완 가이드 생성 중 오류 발생: {e}"
                        st.session_state.feedback_status_msg = ""
                        st.rerun()
            else:
                st.session_state.feedback_error_msg = "❌ 신규 질문, 기존 가이드 초안, 피드백 내용이 모두 입력되어야 합니다."
                st.session_state.feedback_status_msg = ""
                st.rerun()
        
        # 버튼 아래 가이드 정보 동적 노출 레이블 파트
        if st.session_state.feedback_status_msg:
            st.info(st.session_state.feedback_status_msg)
        if st.session_state.feedback_error_msg:
            st.error(st.session_state.feedback_error_msg)
                
    with col_reg_btn:
        st.markdown("<div style='padding-top: 25px;'></div>", unsafe_allow_html=True) # 줄 맞춤 공백용
        
        # 6. 우측 열에 FAQ 등록 버튼 배치 및 물리 디스크 저장 트랜잭션 정상화 패치
        if st.button("✅ 이 가이드를 FAQ 지식으로 최종 채택 및 등록", use_container_width=True, key="btn_faq_final_submit"):
            if admin_question.strip() and final_answer_text.strip():
                if "1. 시스템 환경" in final_answer_text and "2. 조치가이드" in final_answer_text:
                    try:
                        vector_store = get_faq_vector_store()
                        doc_id = str(uuid.uuid4())
                        
                        # 지식 자산 고유 적재 트랜잭션 시행 (질문 접두어 전면 배제 및 순수 본문만 저장)
                        vector_store.add_texts(
                            texts=[final_answer_text.strip()],
                            metadatas=[{
                                "source_file": "ADMIN_APPROVED_FAQ",
                                "category": "FAQ_PLAYBOOK",
                                "approved_question": admin_question.strip()
                            }],
                            ids=[doc_id]
                        )
                        # 팝업 알림 송출 (토스트 알림 및 전체 인풋 폼 세션 완전 클리어 초기화)
                        st.toast("🎉 [등록 성공] 지정 표준 포맷 무결성 검증을 통과하여 순수 지식 자산이 안정적으로 등록되었습니다!", icon="✅")
                        
                        # 전체 입력창 버퍼 초기화 (락 에러 전면 방어)
                        st.session_state.form_question = ""
                        st.session_state.form_feedback = ""
                        st.session_state.admin_ai_draft = ""
                        st.session_state.feedback_status_msg = ""
                        st.session_state.feedback_error_msg = ""
                        
                        st.success("🎉 물리 자산 적재 완료 및 입력창 초기화 성공!")
                        st.rerun()
                    except Exception as ex:
                        st.toast(f"❌ Chroma DB FAQ 자산고 디스크 기입 중 치명적 오류 발생: {ex}", icon="🚨")
                        st.error(f"Chroma DB FAQ 자산고 물리 디스크 기입 실패: {ex}")
                else:
                    st.toast("❌ 필수 가이드 표준 서식이 유실되었습니다.", icon="⚠️")
                    st.error("❌ [포맷 가드레일 위반] 필수 출력 표준 규격(1. 시스템 환경, 2. 조치가이드)이 누락되었습니다. 서식을 확인하세요.")
            else:
                st.toast("⚠️ 등록할 내용이 누락되었습니다.", icon="⚠️")
                st.error("등록할 질문 원문과 최종 편집 공간의 답변 내용이 비어있습니다.")

# ====================================================================
# [Tab 2] 기존 적재 FAQ 자산 통합 조회 및 관리 (CRUD - 수정, 삭제)
# ====================================================================
with tab2:
    st.header("🗂️ 로컬 지식 자산고 대시보드 및 실시간 데이터 무결성 CRUD")
    
    try:
        vector_store = get_faq_vector_store()
        collection_data = vector_store._collection.get()
        
        if collection_data and "ids" in collection_data and collection_data["ids"]:
            faq_ids = collection_data["ids"]
            faq_documents = collection_data["documents"]
            faq_metadatas = collection_data["metadatas"]
            
            display_options = []
            id_map = {}
            
            for idx, meta in enumerate(faq_metadatas):
                q_text = meta.get("approved_question", f"무명 FAQ 자산 ({faq_ids[idx][:8]})")
                option_str = f"[{idx+1}] {q_text}"
                display_options.append(option_str)
                id_map[option_str] = {
                    "id": faq_ids[idx],
                    "question": q_text,
                    "document": faq_documents[idx],
                    "metadata": meta
                }
                
            selected_option = st.selectbox("수정 또는 삭제 처리할 대상 FAQ 문항 선택:", display_options, key="crud_selectbox_target")
            
            if selected_option:
                target_asset = id_map[selected_option]
                st.markdown(f"**📌 타겟 자산 내부 고유 식별 ID:** `{target_asset['id']}`")
                
                mod_question = st.text_input("💡 질문 명세 편집 변경:", value=target_asset["question"], key="crud_mod_question_input")
                mod_document = st.text_area("📋 조치 내용 본문 전체 가이드 편집 (질문 태그 제외 순수 본문):", value=target_asset["document"], height=250, key="crud_mod_document_input")
                
                manage_col1, manage_col2 = st.columns(2)
                
                # 🔄 [UPDATE] 자산 수정 로직 (선삭제 후적재 트랜잭션)
                with manage_col1:
                    if st.button("🔄 선택한 FAQ 데이터 형상 업데이트 반영", key="crud_update_action_btn"):
                        if mod_question.strip() and mod_document.strip():
                            if "1. 시스템 환경" in mod_document and "2. 조치가이드" in mod_document:
                                try:
                                    # 1단계: 기존 파괴적 격리 삭제
                                    vector_store._collection.delete(ids=[target_asset["id"]])
                                    
                                    # 2단계: 최신 형상 재주입 (질문 접두어를 배제하고 수정 본문만 기입)
                                    vector_store.add_texts(
                                        texts=[mod_document.strip()],
                                        metadatas=[{
                                            "source_file": "ADMIN_APPROVED_FAQ",
                                            "category": "FAQ_PLAYBOOK",
                                            "approved_question": mod_question.strip()
                                        }],
                                        ids=[target_asset["id"]]
                                    )
                                    st.toast("🔄 데이터 형상 동기화 성공!", icon="✅")
                                    st.rerun()
                                except Exception as ex:  # 🔥 [교정 마감] catch -> except 문법 오류 완벽 치유
                                    st.error(f"Chroma 수정 트랜잭션 오류 발생: {ex}")
                            else:
                                st.error("❌ [수정 포맷 위반] 필수 출력 표준 규격 서식이 유실되었습니다.")
                        else:
                            st.error("공백 문자 덤프 적재는 허용되지 않습니다.")
                            
                # 🗑️ [DELETE] 자산 삭제 로직
                with manage_col2:
                    if st.button("🗑️ 선택한 FAQ 데이터 자산 영구 삭제", key="crud_delete_action_btn"):
                        try:
                            vector_store._collection.delete(ids=[target_asset["id"]])
                            st.toast("🗑️ 로컬 Chroma DB에서 자산이 영구 삭제되었습니다.", icon="🗑️")
                            st.rerun()
                        except Exception as ex:  # 🔥 [교정 마감] catch -> except 문법 오류 완벽 치유
                            st.error(f"Chroma 삭제 트랜잭션 오류 발생: {ex}")
        else:
            st.info("💡 현재 로컬 Chroma DB 물리 저장소에 영구 누적 적재된 맞춤형 FAQ 플레이북 자산이 없습니다.")
    except Exception as main_e:
        st.info("지식 저장소 동기화 중 세션 준비 대기 상태입니다.")