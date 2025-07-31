# import uuid
# import traceback
# import services.nlu_service as nlu

# from sqlalchemy.ext.asyncio import AsyncSession
# from langchain_openai import ChatOpenAI

# from models.schemas import (
#     ChatResponse,
#     UserRequestInfo,
#     FinalRecommendation,
#     StartChatResponse,
#     HistorySummary
# )
# from typing import List
# from datetime import datetime
# from . import db_service, filter_service, crawl_service
# from core.config import settings


# # --- 핵심 로직: 새로운 채팅 시작 및 첫 메시지 처리 ---

# async def start_new_chat_session(user_id: str, initial_message: str, db: AsyncSession) -> StartChatResponse:
#     """
#     새로운 채팅 세션을 만들고, 첫 메시지가 있으면 바로 처리까지 합니다.
#     """
#     session_id = str(uuid.uuid4())
#     # 초기 state 객체를 생성합니다.
#     state = {
#         "session_id": session_id,
#         "user_id": user_id,
#         "conversation_history": [],
#         "user_info": UserRequestInfo().model_dump()
#     }

#     # 첫 메시지가 있다면, process_chat_message 함수를 바로 호출하여 처리합니다.
#     if initial_message:
#         print(f"[{session_id}] 새로운 채팅 세션 시작과 함께 첫 메시지 처리: {initial_message}")
#         chat_response = await process_chat_message(state, initial_message, db)
        
#         # StartChatResponse 모델에 맞게 데이터를 포장하여 반환합니다.
#         return StartChatResponse(
#             state=chat_response.state,
#             bot_response=chat_response.response
#         )
    
#     # (만약의 경우) 첫 메시지가 없다면, 기본 인사말과 함께 상태를 반환합니다.
#     else:
#         initial_bot_response = "안녕하세요! 무엇을 도와드릴까요?"
#         state["conversation_history"].append(f"Bot: {initial_bot_response}")
#         return StartChatResponse(state=state, bot_response=initial_bot_response)

# # --- 핵심 로직: 메시지 처리 및 대화 진행 ---

# async def process_chat_message(state: dict, user_input: str, db: AsyncSession) -> ChatResponse:
#     """
#     진행 중인 대화에서 사용자의 메시지를 처리하고 다음 응답을 생성합니다.
#     """
#     # state에서 user_info를 객체로 복원하고 session_id를 가져옵니다.
#     user_info = UserRequestInfo(**state.get("user_info", {}))
#     session_id = state.get("session_id")

#     # 1. LLM을 사용하여 사용자 정보 업데이트
#     await _update_user_info(user_info, user_input)

#     # 2. 대화 기록 및 상태 업데이트
#     state["conversation_history"].append(f"User: {user_input}")
#     state["user_info"] = user_info.model_dump()

#     # 3. 필수 정보가 모두 수집되었는지 확인
#     if not user_info.is_ready():
#         next_question = _get_next_question(user_info)
#         state["conversation_history"].append(f"Bot: {next_question}")
#         return ChatResponse(
#             response=next_question,
#             session_id=session_id,
#             state=state  # 업데이트된 state를 반드시 포함
#         )

#     # 4. 모든 정보 수집 완료 -> 추천 프로세스 시작
#     try:
#         print(f"[{session_id}] 모든 정보 수집 완료. 추천 프로세스 시작.")
#         print(f"[{session_id}] 사용자 요청: {user_info.model_dump_json(indent=2)}")

#         # 4-1. DB에서 위치 기반 후보군 조회
#         restaurants_step1 = await db_service.get_restaurants_from_db(db, user_info.location, user_info.amenities)
#         if not restaurants_step1:
#             return ChatResponse(response="해당 조건에 맞는 음식점이 없습니다. 다른 지역으로 다시 시도해보세요.", session_id=session_id, is_final=True, state=state)

#         # 4-2. 후보 음식점들의 정규화된 메뉴 조회
#         menus_by_restaurant = await db_service.get_normalized_menus_for_restaurants(db, restaurants_step1)
#         all_menus = {menu for menus in menus_by_restaurant.values() for menu in menus}

#         # 4-3. 건강/식단 제약으로 메뉴 필터링
#         suitable_menus = await filter_service.filter_menus_by_health(all_menus, user_info.disease, user_info.dietary_restrictions)
#         if not suitable_menus:
#              return ChatResponse(response="고객님의 건강 조건에 맞는 메뉴를 찾지 못했습니다.", session_id=session_id, is_final=True, state=state)

#         # 4-4. 적합 메뉴를 판매하는 음식점만 남기기
#         restaurants_step4 = [
#             r for r in restaurants_step1 
#             if any(menu in suitable_menus for menu in menus_by_restaurant.get(f"{r['name']} {r.get('branch_name', '')}".strip(), set()))
#         ]
#         if not restaurants_step4:
#             return ChatResponse(response="건강에 좋은 메뉴를 파는 음식점을 찾지 못했습니다.", session_id=session_id, is_final=True, state=state)

#         # 4-5. 기타 요청사항(분위기 등)으로 리뷰 기반 필터링
#         restaurants_step5 = await filter_service.filter_restaurants_by_review(restaurants_step4, user_info.other_requests)

#         # 4-6. 실시간 정보 확인
#         crawled_infos = await crawl_service.get_restaurants_realtime_info(restaurants_step5[:10])
        
#         # 4-7. 영업 여부 최종 판단
#         visitable_restaurants = []
#         for info in crawled_infos:
#             if await crawl_service.check_visitable(info, user_info.time):
#                 visitable_restaurants.append(FinalRecommendation(**info))

#         if not visitable_restaurants:
#             return ChatResponse(response="후보 맛집은 찾았지만, 아쉽게도 요청하신 시간에 영업 중인 곳이 없네요.", session_id=session_id, is_final=True, state=state)

#         # 최종 추천 결과 반환
#         bot_response = "🎉 고객님을 위한 최종 맛집 추천 목록입니다!"
#         state["conversation_history"].append(f"Bot: {bot_response}")

#         return ChatResponse(
#             response=bot_response,
#             session_id=session_id,
#             is_final=True,
#             recommendations=visitable_restaurants[:5],
#             state=state
#         )
#     except Exception as e:
#         print(f"[Orchestrator Error] {e}")
#         traceback.print_exc()
#         return ChatResponse(response="추천 과정 중 오류가 발생했습니다. 다시 시도해주세요.", session_id=session_id, is_final=True, state=state)

# # --- 보조 함수: LLM 정보 추출 ---

# async def _update_user_info(user_info: UserRequestInfo, message: str):
#     """LLM을 사용해 사용자 메시지에서 정보를 추출하여 상태를 업데이트"""
#     llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
#     structured_llm = llm.with_structured_output(UserRequestInfo)
    
#     prompt = f"""
#     아래는 현재까지 수집된 사용자 정보와 새로운 사용자 메시지입니다.
#     메시지를 분석하여 UserRequestInfo JSON 객체를 업데이트해주세요.
#     기존에 값이 있던 필드는 유지하고, 새로운 정보만 추가하거나 수정하세요.

#     [지침]
#     - '강남', '홍대' 등은 '강남구', '마포구'와 같이 '구' 단위로 변환하세요.
#     - amenities: '주차', '와이파이', '놀이방'과 같이 물리적으로 존재하는 편의시설만 포함하세요.
#     - other_requests: '혼밥하기 좋은', '분위기가 조용한', '가성비 좋은' 등과 같이 분위기나 상황, 추상적인 요청을 포함하세요.

#     [기존 정보]
#     {user_info.model_dump_json()}

#     [새 메시지]
#     {message}
#     """
    
#     updated_info = await structured_llm.ainvoke(prompt)
    
#     # 기존 정보 위에 새로 추출된 정보 덮어쓰기 (None이 아닌 값만)
#     for field, value in updated_info.model_dump().items():
#         if value is not None and (isinstance(value, list) and value or not isinstance(value, list)):
#             setattr(user_info, field, value)

# # --- 보조 함수: 다음 질문 생성 ---

# def _get_next_question(user_info: UserRequestInfo) -> str:
#     """다음에 물어봐야 할 질문을 반환"""
#     if not user_info.location: return "어느 지역(구 단위)의 음식점을 찾으시나요?"
#     if not user_info.time: return "언제 방문하실 예정인가요? (예: 오늘 저녁 7시)"
#     if not user_info.dietary_restrictions: return "특별히 피해야 할 음식이나 식단(채식 등)이 있으신가요? (없으면 '없음')"
#     if not user_info.disease: return "혹시 앓고 계신 질환이 있으신가요? (없으면 '없음')"
#     if not user_info.other_requests: return "선호하는 분위기나 '주차', '놀이방' 같은 편의시설 요청이 있으신가요? (없으면 '없음')"
#     return "알 수 없는 상태입니다. 다시 시작해주세요."

# # --- 히스토리 관련 함수들 (추후 DB 연동 필요) ---

# async def get_chat_history(session_id: str, db: AsyncSession) -> dict:
#     # TODO: 실제 프로덕션에서는 이 함수를 DB와 연동해야 합니다.
#     # 지금은 state가 모든 정보를 가지고 있으므로, 클라이언트가 가진 state를 그대로 사용하는 것이 더 효율적입니다.
#     # 이 함수는 예시로 남겨둡니다.
#     print(f"DB에서 {session_id} 기록 조회 시도 (현재는 구현되지 않음)")
#     raise NotImplementedError("채팅 기록 조회 기능은 아직 DB와 연동되지 않았습니다.")


# async def get_user_history(user_id: str, db: AsyncSession) -> List[HistorySummary]:
#     # TODO: 실제 프로덕션에서는 DB에서 user_id에 해당하는 모든 세션의 요약 정보를 조회해야 합니다.
#     print(f"DB에서 {user_id}의 모든 채팅 목록 조회 시도 (현재는 구현되지 않음)")
#     # 임시로 빈 목록 반환
#     return []


# services/chat_orchestrator.py

import uuid
import traceback
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

# 새로운 스키마에 맞게 import 구문 수정
from models.schemas import ChatResponse, UserRequestInfo, FinalRecommendation, StartChatResponse, HistorySummary
from . import db_service, filter_service, crawl_service, nlu_service

# --- 핵심 로직: 새로운 채팅 시작 및 첫 메시지 처리 ---
async def start_new_chat_session(user_id: str, initial_message: Optional[str], db: AsyncSession) -> StartChatResponse:
    session_id = str(uuid.uuid4())
    state = {
        "session_id": session_id,
        "user_id": user_id,
        "conversation_history": [],
        "user_info": UserRequestInfo().model_dump()
    }

    if initial_message:
        print(f"[{session_id}] 새 채팅 시작과 함께 첫 메시지 처리: {initial_message}")
        chat_response = await process_chat_message(state, initial_message, db)
        return StartChatResponse(
            state=chat_response.state,
            bot_response=chat_response.response
        )
    else:
        initial_bot_response = "안녕하세요! 무엇을 도와드릴까요?"
        state["conversation_history"].append(f"Bot: {initial_bot_response}")
        await db_service.log_interaction(db, session_id, "chat_start", "bot_response", initial_bot_response, user_id)
        return StartChatResponse(state=state, bot_response=initial_bot_response)

# --- 핵심 로직: 메시지 처리 및 대화 진행 ---
async def process_chat_message(state: dict, user_input: str, db: AsyncSession) -> ChatResponse:
    user_info = UserRequestInfo(**state.get("user_info", {}))
    session_id = state.get("session_id")
    user_id = state.get("user_id")

    # 1. 대화 기록 및 정보 업데이트
    state["conversation_history"].append(f"User: {user_input}")
    await db_service.log_interaction(db, session_id, "user_interaction", "user_input", user_input, user_id)

    await _update_user_info(user_info, user_input, state["conversation_history"])
    state["user_info"] = user_info.model_dump()

    # 2. 필수 정보 수집 여부 확인
    if not user_info.is_ready():
        next_question = _get_next_question(user_info)
        state["conversation_history"].append(f"Bot: {next_question}")
        await db_service.log_interaction(db, session_id, "user_interaction", "bot_response", next_question, user_id)
        return ChatResponse(response=next_question, session_id=session_id, state=state)

    # 3. 모든 정보 수집 완료 -> 추천 프로세스 시작
    try:
        print(f"[{session_id}] 모든 정보 수집 완료. 추천 프로세스 시작.")
        print(f"    - 최종 수집 정보: {user_info.model_dump_json(indent=2)}")

        # [수정] 1. 지역명을 '구' 단위로 변환
        print(f"    - 원본 위치 입력: '{user_info.location}'")
        location_gu = await nlu_service.extract_gu_from_location(user_info.location)
        if location_gu == "알 수 없음":
            print(f"[DEBUG] 종료: 위치 '{user_info.location}'에서 '구'를 추출하지 못했습니다.")
            return ChatResponse(response=f"'{user_info.location}' 지역을 정확히 이해하지 못했어요. '강남구', '송파구' 처럼 다시 말씀해주시겠어요?", session_id=session_id, is_final=True, state=state)
        print(f"    - 변환된 위치 (구): '{location_gu}'")

        # [수정] 2. 변환된 지역명(location_gu)을 DB 조회에 사용
        restaurants_step1 = await db_service.get_restaurants_from_db(db, location_gu, user_info.amenities)
        if not restaurants_step1:
            print("[DEBUG] 종료: 1단계에서 후보 음식점을 찾지 못했습니다.")
            return ChatResponse(response="해당 조건에 맞는 음식점이 없습니다.", session_id=session_id, is_final=True, state=state)
        print(f"[DEBUG] 1단계 통과. 후보: {len(restaurants_step1)}개")

        menus_by_restaurant = await db_service.get_normalized_menus_for_restaurants(db, restaurants_step1)
        all_menus = {menu for menus in menus_by_restaurant.values() for menu in menus}
        print(f"[DEBUG] 2단계 통과. 전체 메뉴 후보: {len(all_menus)}개")

        # [수정] 3. 질병명 정규화
        print(f"    - 원본 질병 입력: '{user_info.disease}'")
        normalized_disease = await nlu_service.normalize_disease_name(user_info.disease)
        print(f"    - 정규화된 질병명: '{normalized_disease}'")

        # [수정] 4. 정규화된 질병명을 필터링에 사용
        suitable_menus = await filter_service.filter_menus_by_health(all_menus, normalized_disease, user_info.dietary_restrictions)
        if not suitable_menus:
            print("[DEBUG] 종료: 3단계(건강 필터링)에서 적합한 메뉴를 찾지 못했습니다.")
            return ChatResponse(response="고객님의 건강 조건에 맞는 메뉴를 찾지 못했습니다.", session_id=session_id, is_final=True, state=state)
        print(f"[DEBUG] 3단계 통과. 건강 메뉴 후보: {len(suitable_menus)}개")

        restaurants_step4 = [r for r in restaurants_step1 if any(menu in suitable_menus for menu in menus_by_restaurant.get(f"{r['name']} {r.get('branch_name', '')}".strip(), set()))]
        if not restaurants_step4:
            print("[DEBUG] 종료: 4단계(메뉴 판매점 필터링)에서 적합한 음식점을 찾지 못했습니다.")
            return ChatResponse(response="건강에 좋은 메뉴를 파는 음식점을 찾지 못했습니다.", session_id=session_id, is_final=True, state=state)
        print(f"[DEBUG] 4단계 통과. 후보: {len(restaurants_step4)}개")

        restaurants_step5 = await filter_service.filter_restaurants_by_review(restaurants_step4, user_info.other_requests)
        print(f"[DEBUG] 5단계 통과 (리뷰 필터링). 후보: {len(restaurants_step5)}개")

        final_recommendations = await crawl_service.get_final_recommendations_with_crawling(restaurants_step5[:10], user_info.time)
        print(f"[DEBUG] 6단계 통과 (크롤링 및 최종 필터링). 최종 후보: {len(final_recommendations)}개")

        if not final_recommendations:
            return ChatResponse(response="후보 맛집은 찾았지만, 아쉽게도 요청하신 시간에 영업 중인 곳이 없네요.", session_id=session_id, is_final=True, state=state)

        bot_response = "🎉 고객님을 위한 최종 맛집 추천 목록입니다!"
        await db_service.log_interaction(db, session_id, "final_result", "success", [rec.model_dump() for rec in final_recommendations], user_id)
        return ChatResponse(response=bot_response, session_id=session_id, is_final=True, recommendations=final_recommendations, state=state)

    except Exception as e:
        print(f"[Orchestrator Error] {e}")
        traceback.print_exc()
        await db_service.log_interaction(db, session_id, "pipeline_error", "error", str(e), user_id)
        return ChatResponse(response="추천 과정 중 오류가 발생했습니다.", session_id=session_id, is_final=True, state=state)

# --- 보조 함수들 ---
async def _update_user_info(user_info: UserRequestInfo, user_input: str, history: List[str]):
    """NLU 서비스를 호출하여 사용자 정보를 업데이트합니다."""
    
    extracted_info: UserRequestInfo
    if len(history) <= 2:
        extracted_info = await nlu_service.extract_initial_request(user_input)
    else:
        last_bot_question = history[-2] if len(history) > 1 else ""
        extracted_info = await nlu_service.extract_info_from_message(history, last_bot_question)
    
    # ▼▼▼▼▼ [수정] 업데이트 로직 강화 ▼▼▼▼▼
    # 추출된 값이 의미가 있을 때만 (None이 아니고, 빈 문자열/리스트가 아닐 때만) 업데이트합니다.
    for field, value in extracted_info.model_dump(exclude_unset=True).items():
        if value: # None, "", [] 등 비어있는 값은 모두 False로 처리되어 업데이트되지 않음
            setattr(user_info, field, value)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def _get_next_question(user_info: UserRequestInfo) -> str:
    """다음에 물어봐야 할 질문을 반환합니다."""
    # ▼▼▼▼▼ [수정] 모든 검사 조건을 'is None'으로 통일하여 일관성 확보 ▼▼▼▼▼
    if user_info.location is None: return "어느 지역의 음식점을 찾으시나요?"
    if user_info.time is None: return "언제 방문하실 예정인가요?"
    if user_info.dietary_restrictions is None: return "특별히 피해야 할 음식이 있나요? (없으면 '없음')"
    if user_info.disease is None: return "앓고 있는 질환이 있으신가요? (없으면 '없음')"
    if user_info.other_requests is None: return "선호하는 분위기나 편의시설이 있나요? (없으면 '없음')"
    return "" # 모든 정보 수집 완료
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

async def get_user_history(user_id: str, db: AsyncSession) -> List[HistorySummary]:
    # TODO: DB에서 user_id에 해당하는 세션 요약 정보 조회
    return []