# import os
# from dotenv import load_dotenv

# # 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 가져오기)
# load_dotenv()

# # OpenAI 임베딩 설정
# from langchain_openai import OpenAIEmbeddings
# embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# # Chroma 서버에 연결
# from chromadb import HttpClient
# client = HttpClient(host="155.248.175.96", port=8000)

# # langchain-chroma로 Chroma 인스턴스 생성
# from langchain_chroma import Chroma
# db = Chroma(
#     client=client,
#     collection_name="disease_and_diet",
#     embedding_function=embedding
# )

# # 전체 조회 후 상위 10개만 출력
# collection = client.get_collection("langchain")
# all_data = collection.get(include=["documents", "metadatas"])

# # 최대 10개만 출력
# max_items = min(10, len(all_data["ids"]))
# for i in range(max_items):
#     print(f"ID: {all_data['ids'][i]}")
#     print(f"Document: {all_data['documents'][i]}")
#     print(f"Metadata: {all_data['metadatas'][i]}")
#     print("-" * 40)


import sqlite3

# 경로를 raw string으로 처리
db_path = r"G:\내 드라이브\AIProject\chroma_db_health_judgments\chroma.sqlite3"

# 연결 시도
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT * FROM embedding_fulltext_search_content LIMIT 100")
rows = cursor.fetchall()

for i, row in enumerate(rows):
    print(f"[{i+1}] Document ID: {row[0]}")
    print(f"Content:\n{row[1]}")
    print("-" * 40)
