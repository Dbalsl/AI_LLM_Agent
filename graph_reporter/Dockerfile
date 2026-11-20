# ==============================================================================
# Google Cloud Run 및 컨테이너 배포용 Dockerfile
# VERITAS // AI Desk - LangGraph 기반 탐사보도 신문기자 백엔드 서버
# ==============================================================================

FROM python:3.11-slim

# 시스템 패키지 업데이트 및 빌드에 필요한 기본 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 파이썬 실시간 로그 출력 및 Cloud Run 기본 포트 환경변수
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 의존성 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY main.py .

# 포트 오픈 및 ASGI(Uvicorn) 서버 실행
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
