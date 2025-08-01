import os
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 가져오기)
load_dotenv()

# OpenAI 임베딩 설정
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Chroma 서버에 연결
from chromadb import HttpClient
client = HttpClient(host="155.248.175.96", port=8000)

# langchain-chroma로 Chroma 인스턴스 생성
from langchain_chroma import Chroma
db = Chroma(
    client=client,
    collection_name="disease_and_diet",
    embedding_function=embedding
)

# 전체 조회 후 상위 10개만 출력
collection = client.get_collection("foods_data")
all_data = collection.get(include=["documents", "metadatas"])

# 최대 10개만 출력
max_items = min(10, len(all_data["ids"]))
for i in range(max_items):
    print(f"ID: {all_data['ids'][i]}")
    print(f"Document: {all_data['documents'][i]}")
    print(f"Metadata: {all_data['metadatas'][i]}")
    print("-" * 40)