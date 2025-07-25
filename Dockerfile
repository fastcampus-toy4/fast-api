# 1) 베이스 이미지
FROM python:3.11-slim

# 2) 작업 디렉토리
WORKDIR /app

# 3) 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 소스 복사
COPY . .

# 5) 컨테이너 시작 명령
CMD ["uvicorn", "langchaintest:app", "--host", "0.0.0.0", "--port", "9000"]
