#!/bin/bash

# 🔥 [보완 반영] 컨테이너 내부 하위 페이지(pages/)에서도 루트의 model_factory를 무조건 찾도록 경로 강제 매핑
export PYTHONPATH=$PYTHONPATH:/app

# 1. 백엔드 FastAPI 서버를 백그라운드(&)로 먼저 실행
echo "[Docker Engine] Starting FastAPI Backend Core..."
python -m backend.main &

# 2. 백엔드가 완전히 살 때까지 3초간 일시 대기
sleep 3

# 3. 프론트엔드 Streamlit 서비스를 메인 프로세스로 기동
echo "[Docker Engine] Starting Streamlit UI Platform..."
streamlit run frontend/app_ui.py --server.port 8501 --server.address 0.0.0.0