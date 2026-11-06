# ==============================================================================
# Google Cloud Run 배포용 Dockerfile
# FAISS + Multi-Query + Hybrid Search + RAG Fusion 백엔드 서버
# ==============================================================================

FROM python:3.11-slim

# 시스템 패키지 업데이트 및 FAISS 구동에 필요한 필수 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 파이썬 출력 버퍼링 해제 (실시간 로그 및 스트리밍 즉시 출력)
ENV PYTHONUNBUFFERED=1
# Cloud Run 기본 포트 설정
ENV PORT=8080

# 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY main.py .

# Cloud Run 컨테이너 포트 오픈 및 Uvicorn 서버 실행
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

