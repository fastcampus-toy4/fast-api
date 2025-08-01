# # core/lifespans.py

# from contextlib import asynccontextmanager
# import pandas as pd
# from fastapi import FastAPI
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_google_community import GoogleSearchAPIWrapper
# from db.database import init_db, get_db_engine
# from core.config import settings

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     FastAPI 애플리케이션의 라이프사이클 동안 실행될 컨텍스트 매니저.
#     서버 시작 시 필요한 리소스를 로드하고, 종료 시 정리합니다.
#     """
#     print("="*50)
#     print("AI 맛집 추천 시스템 초기화를 시작합니다...")

#     # --- 1. 설정 및 전역 변수 초기화 ---
#     app.state.settings = settings
#     app.state.llm = ChatOpenAI(model="gpt-4o", temperature=0)
#     app.state.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
#     app.state.google_search = GoogleSearchAPIWrapper()
#     app.state.prompt_cache = {}

#     # --- 2. DB 엔진 초기화 ---
#     app.state.db_engine = init_db(settings)
#     if app.state.db_engine:
#         print("✅ 데이터베이스 엔진이 성공적으로 초기화되었습니다.")
#     else:
#         print("❌ 데이터베이스 엔진 초기화에 실패했습니다.")
    
#     # --- 3. 사전 계산된 데이터 및 Vector DB 로드 ---
#     try:
#         print("사전 계산된 클러스터 및 건강 판단 데이터를 로딩합니다...")
#         df_clusters = pd.read_csv(settings.FOOD_CLUSTERS_PATH)
#         app.state.food_to_cluster_map = pd.Series(df_clusters.cluster_id.values, index=df_clusters.food_name).to_dict()

#         df_representatives = pd.read_csv(settings.REPRESENTATIVE_FOODS_PATH)
#         app.state.cluster_to_food_map = pd.Series(df_representatives.representative_food.values, index=df_representatives.index).to_dict()

#         app.state.health_judgment_db = Chroma(
#             persist_directory=settings.CHROMA_DB_HEALTH_JUDGMENTS_PATH,
#             embedding_function=app.state.embedding_function
#         )
#         app.state.review_db = Chroma(
#             persist_directory=settings.CHROMA_DB_REVIEW_PATH,
#             embedding_function=app.state.embedding_function
#         )
#         print("✅ 데이터 로딩 완료.")
#     except Exception as e: # FileNotFoundError 대신 모든 에러(Exception)를 잡도록 변경
#         print("="*50)
#         print(f"[치명적 시작 오류] 애플리케이션 초기화 중 에러 발생: {e}")
#         import traceback
#         traceback.print_exc() # 오류의 모든 상세 내용을 출력!
#         print("="*50)
#         # 여기서 sys.exit(1) 등으로 서버를 강제 종료할 수도 있습니다.

#     print("="*50)
#     print("🚀 시스템이 성공적으로 시작되었습니다.")
#     yield
    
#     # --- 애플리케이션 종료 시 실행될 코드 ---
#     if app.state.db_engine:
#         app.state.db_engine.dispose()
#         print("데이터베이스 연결이 종료되었습니다.")
#     print("👋 시스템이 종료되었습니다.")

# core/lifespans.py
# core/lifespans.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from langchain_openai import ChatOpenAI

# services 폴더의 data_loader를 사용하도록 수정합니다.
from services import data_loader
from core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 라이프사이클(시작/종료)을 관리합니다.
    서버 시작 시 AI 모델과 필요한 데이터를 메모리에 올립니다.
    """
    print("="*50)
    print("AI 맛집 추천 시스템 초기화를 시작합니다...")

    # --- 1. AI 모델(LLM) 초기화 ---
    # app.state에 llm 객체를 만들어 다른 파일에서 공유할 수 있도록 합니다.
    app.state.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.***REMOVED***)
    print("✅ AI 모델(LLM)이 성공적으로 초기화되었습니다.")

    # --- 2. 사전 데이터 로드 ---
    try:
        data_loader.load_all_data()
        print("✅ 사전 데이터가 성공적으로 메모리에 로드되었습니다.")
    except Exception as e:
        print(f"🔥 [오류] 서버 시작 실패: 데이터 로딩 중 문제가 발생했습니다.")
        print(f"    - 오류 내용: {e}")
        import traceback
        traceback.print_exc()

    print("="*50)
    print("🚀 시스템이 성공적으로 시작되었습니다.")
    
    yield
    
    # --- 애플리케이션 종료 시 실행될 코드 ---
    print("👋 시스템이 종료되었습니다.")