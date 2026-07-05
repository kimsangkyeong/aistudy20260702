# 1. 사내망과 완벽히 일치하는 파이썬 3.12 표준 베이스 이미지 지정
FROM python:3.12-slim

# 2. 컨테이너 내부 작업 디렉토리 빌드
WORKDIR /app

# 3. 필수 OS 보안 컴파일러 레이어 선언 (Chroma 및 FlashRank 컴파일 방어)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 소스 카피 및 캐시 레이어 활용 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 전체 소스 코드 코드 이관
COPY . .

# 6. FastAPI(8000) 및 Streamlit(8501) 포트 전역 개방
EXPOSE 8000
EXPOSE 8501
 
# 실행 전 윈도우/리눅스 개행 문자 충돌 방지 및 가동 권한 부여 선언
RUN apt-get update && apt-get install -y dos2unix && \
    dos2unix start.sh && \
    chmod +x start.sh

# 🔥 [최종 완결] 단일 명령어가 아닌 통합 가동 쉘 스크립트를 엔트리포인터로 지정
CMD ["./start.sh"]