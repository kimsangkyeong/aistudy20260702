# 실행절차

## Step 1. 패키기 설
### [터미널 1] 누락된 의존성 패키지 클린 재설치 :
```bash
pip install -r requirements.txt

## Step 2. 서버기동하기
### [터미널 1] 고도화된 REST API 백엔드 실행:
```bash
python -m backend.main

## Step 3. UI 기동하기
### [터미널 2] 입력 방어선이 결합된 Streamlit UI 구동:
```bash
streamlit run frontend/app_ui.py