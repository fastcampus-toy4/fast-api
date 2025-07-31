from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

# --- 내부 로직용 Pydantic 모델 ---

class UserRequestInfo(BaseModel):
    """사용자의 요청 정보를 담는 핵심 모델"""
    location: Optional[str] = None
    time: Optional[str] = None
    dietary_restrictions: Optional[str] = None
    disease: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    other_requests: Optional[str] = None

    def is_ready(self) -> bool:
        """추천을 시작하기에 모든 필수 정보가 채워졌는지 확인"""
        return all([
            self.location,
            self.time,
            self.dietary_restrictions is not None,
            self.disease is not None,
            self.other_requests is not None
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


# --- API 요청/응답 모델 ---

class FinalRecommendation(BaseModel):
    """최종 추천 맛집 정보 모델"""
    restaurant_full_name: str
    address: Optional[str] = None
    operating_hours: Optional[str] = None
    phone_number: Optional[str] = None
    holiday_info: Optional[str] = None

class StartChatRequest(BaseModel):
    """채팅 시작 시 요청 모델"""
    user_id: str
    initial_message: Optional[str] = None

class StartChatResponse(BaseModel):
    """채팅 시작 시 응답 모델"""
    state: dict
    bot_response: str

class ChatRequest(BaseModel):
    """채팅 메시지 전송 시 요청 모델"""
    state: dict
    user_input: str

class ChatResponse(BaseModel):
    """채팅 메시지 전송 시 응답 모델"""
    response: str
    session_id: str
    is_final: bool = False
    recommendations: Optional[List[FinalRecommendation]] = None
    state: dict

class HistorySummary(BaseModel):
    """채팅 기록 요약 목록을 위한 모델"""
    session_id: str
    title: str
    created_at: datetime

    
