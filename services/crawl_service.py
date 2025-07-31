import json # <--- 이 한 줄을 추가하세요!
import asyncio

# services/crawl_service.py
from typing import Dict, List
from langchain_openai import ChatOpenAI
from models.schemas import CrawledInfo # Pydantic 모델 사용
from core.config import settings


async def get_restaurants_realtime_info(restaurants: List[Dict]) -> List[Dict]:
    """
    ⚠️ 중요: 이 함수는 실제 크롤링 로직의 '플레이스홀더'입니다.
    Selenium과 같은 블로킹 I/O 라이브러리를 FastAPI의 메인 프로세스에서 직접 실행하면
    서버 전체 성능이 심각하게 저하됩니다. (이벤트 루프 차단)

    [권장 해결책]
    1. Celery 또는 ARQ와 같은 분산 태스크 큐(Task Queue)를 도입합니다.
    2. FastAPI는 크롤링 작업을 이 큐에 전달하고 즉시 사용자에게 응답합니다.
    3. 별도의 '워커(Worker)' 프로세스가 큐에서 작업을 가져와 Selenium으로 크롤링을 수행하고 결과를 DB에 저장합니다.
    4. 클라이언트는 나중에 작업 상태를 폴링하거나 WebSocket으로 결과를 수신합니다.

    아래 코드는 이러한 비동기 아키텍처를 시뮬레이션한 예시입니다.
    """
    print(f"백그라운드 크롤링 시뮬레이션 시작 ({len(restaurants)}곳 대상)...")
    
    # 실제로는 Task Queue에 작업을 보내고, 여기서는 가상 결과를 반환합니다.
    # 각 음식점의 크롤링이 성공했다고 가정하고 임시 데이터를 채웁니다.
    crawled_results = []
    for r in restaurants:
        full_name = f"{r['name']} {r.get('branch_name', '')}".strip()
        # 가상 크롤링 결과 생성
        crawled_data = {
            "restaurant_full_name": full_name,
            "address": "서울시 어딘가 (크롤링 필요)",
            "operating_hours": "매일 11:00 - 22:00 (크롤링 필요)",
            "phone_number": "02-1234-5678 (크롤링 필요)",
            "holiday_info": "연중무휴 (크롤링 필요)",
        }
        crawled_results.append(crawled_data)

    print("백그라운드 크롤링 시뮬레이션 완료.")
    return crawled_results


async def check_visitable(info: Dict, user_time: str) -> bool:
    """
    크롤링된 영업시간 정보와 사용자의 방문 희망 시간을 바탕으로
    LLM을 이용해 현재 방문 가능한지 여부를 판단합니다.
    """
    # model_kwargs를 사용하여 JSON 모드 활성화
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0, 
        api_key=settings.OPENAI_API_KEY,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )
    
    prompt = f"""
    아래 음식점 정보와 사용자 방문 희망 시간을 보고, 현재 방문이 가능한지 판단해주세요.
    "visitable" 필드에 true 또는 false 값만 포함하는 JSON 형식으로만 답변해주세요.
    어떠한 설명도 추가하지 말고 JSON 객체만 응답해야 합니다.

    [음식점 정보]
    - 영업시간: {info.get('opening_hours', '정보 없음')}

    [방문 희망 시간]
    - {user_time}
    """
    
    try:
        response = await llm.ainvoke(prompt)
        # LLM의 응답(문자열)을 JSON 객체로 파싱
        result_json = json.loads(response.content)
        return result_json.get("visitable", False)
    except Exception as e:
        print(f"영업시간 판단 LLM 오류: {e}")
        # 오류 발생 시 안전하게 '방문 불가'로 처리
        return False