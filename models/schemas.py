# models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# --- API 요청/응답 모델 ---

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None # 대화의 연속성을 위한 세션 ID

class FinalRecommendation(BaseModel):
    name: str
    branch_name: Optional[str] = None
    category: str
    address: str
    phone_number: Optional[str] = None
    opening_hours: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    visitable_now: bool = False
    reason: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    is_final: bool = False
    recommendations: Optional[List[FinalRecommendation]] = None
    state: Optional[dict] = None # state 필드 확인

# --- 내부 로직용 Pydantic 모델 ---

class UserRequestInfo(BaseModel):
    location: Optional[str] = None
    time: Optional[str] = None
    dietary_restrictions: Optional[str] = None
    disease: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    other_requests: Optional[str] = None

    def is_ready(self) -> bool:
        """
        모든 필수 정보가 수집되었는지 확인하는 함수.
        [수정] other_requests가 채워졌는지 확인하는 로직을 추가합니다.
        """
        return all([
            self.location,
            self.time,
            self.dietary_restrictions,
            self.disease,
            self.other_requests is not None # 'None'이 아니어야만 통과
        ])
        
# ▼▼▼▼▼ 누락되었던 SeoulGuInfo 모델을 여기에 추가합니다 ▼▼▼▼▼
class SeoulGuInfo(BaseModel):
    gu_name: str = Field(description="서울의 25개 행정구 중 가장 관련성 높은 '구'의 이름")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

class CrawledInfo(BaseModel):
    """크롤링된 텍스트에서 추출한 정보 모델"""
    address: Optional[str] = Field(None, description="추출된 주소")
    operating_hours: Optional[str] = Field(None, description="추출된 영업시간")
    phone_number: Optional[str] = Field(None, description="추출된 전화번호")
    holiday_info: Optional[str] = Field(None, description="추출된 휴무일 정보")

class StartChatRequest(BaseModel):
    user_id: str
    initial_message: Optional[str] = None

class StartChatResponse(BaseModel):
    state: dict
    bot_response: str

class ChatMessageRequest(BaseModel):
    state: dict
    user_input: str

class HistorySummary(BaseModel):
    session_id: str
    content: str
    created_at: datetime
class StartChatRequest(BaseModel):
    user_id: str
    # [중요] 사용자의 첫 메시지를 받을 수 있도록 이 항목을 추가합니다.
    # optional 필드로 만들기 위해 기본값을 None으로 설정합니다.
    initial_message: str | None = None

