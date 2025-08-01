# services/data_loader.py

import pandas as pd
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from core.config import settings

# --- 전역 변수: 로드된 데이터를 메모리에 저장 ---
HEALTH_JUDGMENT_DB = None
REVIEW_DB = None
FOOD_TO_CLUSTER_MAP = {}
CLUSTER_TO_FOOD_MAP = {}

def load_all_data():
    """
    서버 시작 시 필요한 모든 데이터를 미리 메모리에 로드하는 함수.
    """
    global HEALTH_JUDGMENT_DB, REVIEW_DB, FOOD_TO_CLUSTER_MAP, CLUSTER_TO_FOOD_MAP

    print("사전 데이터 로드를 시작합니다...")
    
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)
    
    try:
        print(f"-> ChromaDB 서버 ({settings.CHROMA_HOST}:{settings.CHROMA_PORT})에 연결합니다...")
        chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        
        # ▼▼▼▼▼ 1. 건강 정보용 컬렉션 이름을 실제 DB 이름으로 수정 ▼▼▼▼▼
        health_collection_name = "langchain"
        print(f"-> ChromaDB ({health_collection_name}) 컬렉션을 불러옵니다...")
        HEALTH_JUDGMENT_DB = Chroma(
            client=chroma_client,
            collection_name=health_collection_name,
            embedding_function=embedding_function
        )

        # ▼▼▼▼▼ 2. 리뷰용 컬렉션 이름을 실제 DB 이름으로 수정 ▼▼▼▼▼
        review_collection_name = "restaurant_review"
        print(f"-> ChromaDB ({review_collection_name}) 컬렉션을 불러옵니다...")
        REVIEW_DB = Chroma(
            client=chroma_client,
            collection_name=review_collection_name,
            embedding_function=embedding_function
        )
        print("-> Vector DB 로드 완료.")
    except Exception as e:
        print(f"🔥 [오류] ChromaDB 서버 연결 또는 컬렉션 로딩 실패: {e}")
        raise e

    # --- CSV 파일 로딩 ---
    try:
        print(f"-> CSV (음식 클러스터) 로딩: {settings.FOOD_CLUSTERS_PATH}")
        df_clusters = pd.read_csv(settings.FOOD_CLUSTERS_PATH)
        FOOD_TO_CLUSTER_MAP = pd.Series(df_clusters.cluster_id.values, index=df_clusters.food_name).to_dict()

        print(f"-> CSV (대표 음식) 로딩: {settings.REPRESENTATIVE_FOODS_PATH}")
        df_representatives = pd.read_csv(settings.REPRESENTATIVE_FOODS_PATH)
        CLUSTER_TO_FOOD_MAP = pd.Series(df_representatives.representative_food.values, index=df_representatives.index).to_dict()
        print("-> CSV 파일 로드 완료.")
    except FileNotFoundError as e:
        print(f"🔥 [오류] 필수 CSV 파일을 찾을 수 없습니다: {e}")
        raise e