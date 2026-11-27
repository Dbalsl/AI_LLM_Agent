# Google Cloud Run 경량 최적화 Dockerfile (Python 3.11-slim)
FROM python:3.11-slim

# 표준 출력 버퍼링 비활성화 (Cloud Run 로깅 최적화)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# 시스템 의존성 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Cloud Run 수신 포트 오픈
EXPOSE 8080

# Cloud Run 실행 커맨드 (os.environ.get("PORT", 8080) 바인딩)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
