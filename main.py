# main.py

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

# --- core, models, services 모듈 임포트 ---
from core.config import settings
from core.lifespans import lifespan # 새로운 startup 방식
from models.schemas import ChatRequest, ChatResponse, StartChatRequest, StartChatResponse, HistorySummary
from services import chat_orchestrator
from db.dependencies import get_db # 새로운 DB 세션 의존성

# --- FastAPI 앱 초기화 ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI 기반 음식점 추천 시스템 API",
    version="1.0.0",
    lifespan=lifespan  # @app.on_event("startup") 대신 lifespan을 사용
)

# --- CORS 미들웨어 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://155.248.175.96",
        "http://155.248.175.96:3000",
        "http://155.248.175.96:8080", # Spring 서버 주소도 포함
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API 엔드포인트(경로) 정의 ---

@app.get("/", tags=["Root"])
def read_root():
    """서버의 동작 상태를 확인하는 기본 엔드포인트입니다."""
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

@app.post("/chat/start", response_model=StartChatResponse, tags=["Chat"])
async def start_chat(request: StartChatRequest, db: AsyncSession = Depends(get_db)):
    """
    새로운 채팅 세션을 시작하고, 사용자의 첫 메시지가 있다면 함께 처리합니다.
    """
    return await chat_orchestrator.start_new_chat_session(
        user_id=request.user_id,
        initial_message=request.initial_message,
        db=db
    )

@app.post("/chat/message", response_model=ChatResponse, tags=["Chat"])
async def post_message(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    진행 중인 대화에서 사용자의 메시지를 받아 처리하고 다음 응답을 반환합니다.
    """
    if not request.user_input:
        raise HTTPException(status_code=422, detail="user_input 필드는 비워둘 수 없습니다.")
        
    return await chat_orchestrator.process_chat_message(
        state=request.state,
        user_input=request.user_input,
        db=db
    )

@app.get("/history/{user_id}", response_model=List[HistorySummary], tags=["History"])
async def get_user_history(user_id: str, db: AsyncSession = Depends(get_db)):
    """
    특정 사용자의 모든 채팅 기록 요약 목록을 조회합니다.
    """
    return await chat_orchestrator.get_user_history(user_id=user_id, db=db)

# --- 서버 실행 코드 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)