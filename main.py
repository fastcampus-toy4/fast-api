# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.security import get_current_user
from models.schemas import ChatRequest, ChatResponse
from services import chat_orchestrator
from db.database import get_db_session

# FastAPI 앱 초기화
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI 기반 음식점 추천 시스템 API",
    version="1.0.0"
)

# CORS 미들웨어 설정 (React 앱의 주소 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://155.248.175.96:3000",
        "http://155.248.175.96:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 토큰 호출
@app.post("/api/auth/login")
def receive_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials
    print("Received JWT from Spring:", token)
    # 검증 로직 추가 가능
    return {"status": "ok", "token": token}


@app.on_event("startup")
async def startup_event():
    # 애플리케이션 시작 시, data_loader 모듈을 임포트하여
    # 데이터가 메모리에 미리 로드되도록 합니다.
    try:
        from services import data_loader
        print("✅ 사전 데이터 로더 모듈이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"🔥 서버 시작 실패: 데이터 로딩 중 치명적 오류 발생. {e}")
        # 이 경우 서버가 정상 작동할 수 없으므로, 필요 시 프로세스를 강제 종료할 수 있습니다.
        # import os; os._exit(1)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def handle_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session),
    # JWT 토큰 검증 (주석 해제하여 활성화)
    # current_user: dict = Depends(get_current_user) 
):
    """
    챗봇과 대화하여 음식점을 추천받는 메인 엔드포인트입니다.
    - `message`: 사용자가 입력한 메시지
    - `session_id`: 대화의 연속성을 위해 클라이언트가 저장하고 보내야 하는 ID. 첫 대화 시에는 비워둡니다.
    """
    # print(f"현재 사용자: {current_user['username']}") # 인증된 사용자 이름 로깅
    
    result = await chat_orchestrator.process_chat_message(
        message=request.message,
        session_id=request.session_id,
        db=db
    )
    return result


@app.post("/chat/start", response_model=StartChatResponse, tags=["Chat"])
async def start_chat(
    request: StartChatRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    새로운 채팅 세션을 시작합니다.
    """
    result = await chat_orchestrator.start_new_chat_session(
        user_id=request.user_id,
        db=db
    )
    return result

@app.post("/chat/message", response_model=ChatResponse, tags=["Chat"])
async def handle_chat_message(
    request: ChatMessageRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    챗봇과 대화하여 음식점을 추천받는 메인 엔드포인트입니다.
    - `state`: 현재 대화의 상태
    - `user_input`: 사용자가 입력한 메시지
    """
    result = await chat_orchestrator.process_chat_message(
        state=request.state,
        user_input=request.user_input,
        db=db
    )
    return result

@app.get("/chat/history/{session_id}", response_model=ChatResponse, tags=["Chat"])
async def get_chat_history(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    특정 채팅 세션의 기록을 조회합니다.
    """
    result = await chat_orchestrator.get_chat_history(session_id=session_id, db=db)
    return result

@app.get("/history/{user_id}", response_model=List[HistorySummary], tags=["Chat"])
async def get_user_history(
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    사용자별 채팅 세션 목록을 조회합니다.
    """
    result = await chat_orchestrator.get_user_history(user_id=user_id, db=db)
    return result


