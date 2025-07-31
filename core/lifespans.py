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
    app.state.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
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