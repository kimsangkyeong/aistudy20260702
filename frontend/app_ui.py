import os
import sys
import streamlit as st
import requests

# [경로 가드레일] BE 아키텍처 호출을 위한 파이썬 시스템 경로 오케스트레이션
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from model_factory import ModelFactory
from langchain_chroma import Chroma

st.set_page_config(page_title="OCP 4.20 운영 지식 플랫폼", page_icon="🚀", layout="wide")

# 대화 연속성 유지를 위한 고유 스레드 가드레일 및 세션 동기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faq_widget_version" not in st.session_state:
    st.session_state.faq_widget_version = 0

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/chat")
DB_PATH = os.path.abspath(os.getenv("CHROMA_DB_PATH", "./chroma_db"))

st.title("🚀 OpenShift 4.20 코어 인프라 자율 운영 플랫폼")
st.caption("Disconnected 폐쇄망 환경 특화 Self-RAG 및 Multi-Agent 교차 검증 기반 기술 지원 레이어")

# ====================================================================
# [사이드바] 관리자 채택 FAQ 동적 바인딩 및 원터치 검색 인프라
# ====================================================================
with st.sidebar:
    st.header("🗂️ 검증 완료 단축 FAQ 플레이북")
    st.markdown("관리자가 사내 검증을 마친 표준 FAQ 자산 목록입니다. 클릭 시 즉각 원문 매칭 가이드가 로드됩니다.")
    
    faq_options = ["선택 안 함"]
    faq_map = {}
    
    try:
        embeddings = ModelFactory.get_embeddings()
        faq_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name="approved_faq_db")
        collection_data = faq_store._collection.get()
        
        if collection_data and "metadatas" in collection_data and collection_data["metadatas"]:
            for idx, meta in enumerate(collection_data["metadatas"]):
                q_text = meta.get("approved_question", "").strip()
                if q_text and q_text not in faq_options:
                    faq_options.append(q_text)
                    faq_map[q_text] = collection_data["documents"][idx]
    except Exception:
        pass

    # 위젯 교착 상태 해제 기반의 동적 selectbox 바인딩
    selected_faq = st.selectbox(
        "빠른 질의 선택 리스트:",
        options=faq_options,
        index=0,
        key=f"faq_selectbox_v_{st.session_state.faq_widget_version}"
    )

st.markdown("---")

# ====================================================================
# 대화 이력 히스토리 렌더링 공간
# ====================================================================
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        if chat["role"] == "user":
            st.markdown(chat["content"])
        else:
            # 🔴 [개성 반영] 실무자를 위해 단답형을 깨부수고 섹션별 풍부한 렌더링 아키텍처 제공
            res = chat["content"]
            
            # 1. 원인 및 요약 분석부
            st.markdown("### 📋 인프라 영향도 및 원인 분석")
            st.info(res.get("summary", "상세 분석중..."))
            
            # 2. 상세 실행 단계 단계별 렌더링
            st.markdown("### 🛠️ 단계별 표준 조치 절차 (Runbook)")
            for step in res.get("steps", []):
                st.markdown(f"- {step}")
                
            # 3. 터미널 스크립트 공간
            if res.get("code_block") and res["code_block"].strip():
                st.markdown("### 💻 실행 명령어 세트 및 YAML Manifest")
                st.code(res["code_block"].strip(), language="yaml" if "api" in res["code_block"] or "kind" in res["code_block"] else "bash")
                
            # 4. 검증 링크 및 문서 출처
            if res.get("references"):
                st.markdown("### 🔗 관련 Red Hat 기술 포털 및 교차 검증 출처")
                for ref in res["references"]:
                    st.markdown(f"-[{ref}]({ref})" if ref.startswith("http") else f"- `{ref}`")

# ====================================================================
# [트랜잭션 제어] 사용자 질의 및 단축 FAQ 이벤트 리스너 레이어
# ====================================================================
user_query = st.chat_input("OCP 4.20 장애 증상, oc 명령어, 혹은 폐쇄망 미러링 질문을 입력하세요...")
trigger_query = ""

if user_query:
    trigger_query = user_query
elif selected_faq != "선택 안 함":
    trigger_query = selected_faq
    # 단축 질의 클릭 후 위젯 버전을 스위칭 파괴하여 selectbox 상태를 0번(선택 안 함)으로 강제 롤백 초기화
    st.session_state.faq_widget_version += 1

if trigger_query:
    # 1. 유저 인터페이스 메시지 즉각 적재
    st.session_state.chat_history.append({"role": "user", "content": trigger_query})
    with st.chat_message("user"):
        st.markdown(trigger_query)
        
    # 2. 비동기 인프라 탐색 엔진 백엔드 통신 트리거
    with st.chat_message("assistant"):
        with st.spinner("다자간 에이전트 협업 및 실스크립트 문법 무결성 검증 루프 가동 중..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"message": trigger_query, "thread_id": "ops_production_session"},
                    timeout=45
                )
                if response.status_code == 200:
                    structured_res = response.json()
                    
                    # 🔴 [실무형 최적화 시각 렌더링 피드백 인프라]
                    st.markdown("### 📋 인프라 영향도 및 원인 분석")
                    st.info(structured_res.get("summary", "분석이 완료되었습니다."))
                    
                    st.markdown("### 🛠️ 단계별 표준 조치 절차 (Runbook)")
                    for step in structured_res.get("steps", []):
                        st.markdown(f"- {step}")
                        
                    if structured_res.get("code_block") and structured_res["code_block"].strip():
                        st.markdown("### 💻 실행 명령어 세트 및 YAML Manifest")
                        st.code(structured_res["code_block"].strip(), language="yaml" if "api" in structured_res["code_block"] or "kind" in structured_res["code_block"] else "bash")
                        
                    if structured_res.get("references"):
                        st.markdown("### 🔗 관련 Red Hat 기술 포털 및 교차 검증 출처")
                        for ref in structured_res["references"]:
                            st.markdown(f"-[{ref}]({ref})" if ref.startswith("http") else f"- `{ref}`")
                            
                    # 최종 세션 대화 이력 영구 결합
                    st.session_state.chat_history.append({"role": "assistant", "content": structured_res})
                    st.rerun()
                else:
                    st.error(f"백엔드 오케스트레이션 엔진 에러 발생 (코드: {response.status_code})")
            except Exception as e:
                st.error(f"REST API 통신 네트워크 제약 장애 발생: {e}")