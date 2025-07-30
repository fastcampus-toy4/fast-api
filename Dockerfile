FROM jaeleedong/my-fastapi-app-deps:latest

WORKDIR /app

# 의존성은 이미 베이스 이미지에 있으니, 코드만 복사
COPY . .

CMD ["uvicorn", "langchaintest:app", "--host", "0.0.0.0", "--port", "9000"]
