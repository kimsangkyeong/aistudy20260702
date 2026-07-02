# aistudy20260702
ai study

# 1. 의존성 패키지 일괄 설치
pip install -r requirements.txt

# 2. RAG 데이터 파이프라인 가동 (Chroma DB 생성)
python ingest_pdf.py

# 3. 설치 및 환경 설정이 완료된 후 앱 구동
streamlit run app_ui.py
