from typing import List
# ▼▼▼▼▼ FullRequestInfo를 UserRequestInfo로 변경합니다 ▼▼▼▼▼
from models.schemas import UserRequestInfo, SeoulGuInfo
from core.config import settings

def get_llm():
    """FastAPI 앱 상태에서 전역 LLM 객체를 가져오는 헬퍼 함수"""
    from main import app
    return app.state.llm

async def extract_initial_request(first_message: str) -> UserRequestInfo:
    """사용자의 첫 메시지에서 한 번에 모든 정보를 추출합니다."""
    structured_llm = get_llm().with_structured_output(UserRequestInfo)

    # ▼▼▼▼▼ amenities 규칙을 매우 구체적으로 수정한 최종 프롬프트 ▼▼▼▼▼
    prompt = f"""당신은 사용자의 음식점 추천 요청을 분석하여 JSON 객체로 변환하는 전문가입니다.

[사용자의 첫 요청 메시지]
"{first_message}"

[추출 규칙 및 정의]
1.  **`amenities` (편의시설)**: 아래 목록에 있는 단어와 정확히 일치하는 경우에만 추출합니다.
    - **허용 목록**: ["주차", "와이파이", "화장실", "배달"]

2.  **`other_requests` (기타 요청)**: `amenities` 허용 목록에 없는 모든 요구사항은 이 필드에 요약하여 넣습니다.
    - 예시: "조용한", "분위기 좋은", "가성비 좋은", "혼밥하기 좋은", "데이트하기 좋은", "뷰가 좋은", "매콤한", "반려동물 동반", "콜키지 프리"

3.  언급되지 않은 정보는 null로 유지하세요.

[중요 학습 예시]
- 사용자 요청: "주차되고 가성비 좋은 곳"
- 올바른 추출: {{ "amenities": ["주차"], "other_requests": "가성비 좋은 곳" }}

- 사용자 요청: "배달되고 반려동물 동반 가능한 곳"
- 올바른 추출: {{ "amenities": ["배달"], "other_requests": "반려동물 동반 가능" }}

- 사용자 요청: "조용하고 와이파이 되는 곳"
- 올바른 추출: {{ "amenities": ["와이파이"], "other_requests": "조용한 곳" }}

위 규칙과 예시를 바탕으로, 사용자의 첫 요청 메시지를 분석하여 JSON으로 변환해주세요.
"""
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    try:
        return await structured_llm.ainvoke(prompt)
    except Exception as e:
        print(f"[NLU 오류] 초기 정보 추출 실패: {e}")
        return UserRequestInfo()

async def extract_info_from_message(conversation_history: List[str], last_bot_question: str) -> UserRequestInfo:
    """대화의 맥락을 바탕으로, AI의 마지막 질문에 대한 답변만 추출합니다."""
    structured_llm = get_llm().with_structured_output(UserRequestInfo)
    history_string = "\n".join(conversation_history)

    prompt = f"""당신은 사용자의 최신 답변에서 AI의 마지막 질문과 관련된 정보만 추출하는 전문가입니다.

[분석 대상 대화]
{history_string}

[AI의 마지막 질문]
{last_bot_question}

[가장 중요한 규칙]
'AI의 마지막 질문'에 대한 사용자의 답변이 "없음", "없어요", "아니요" 등 명백한 부정의 의미라면, 반드시 해당 필드를 **문자열 "없음"**으로 채워야 합니다.

[추가 지침]
- 'AI의 마지막 질문'과 관련 없는 정보는 추출하지 마세요.
- 답변에 없는 정보는 null로 유지하세요.
"""
    try:
        return await structured_llm.ainvoke(prompt)
    except Exception as e:
        print(f"[NLU 오류] 대화 중 정보 추출 실패: {e}")
        return UserRequestInfo()

async def extract_gu_from_location(location: str) -> str:
    """사용자가 언급한 위치에서 서울시 '구' 이름을 추출합니다."""
    structured_llm = get_llm().with_structured_output(SeoulGuInfo)
    prompt = f"""당신은 서울 지리에 매우 능숙한 전문가입니다. 
사용자의 문장에서 가장 관련성이 높은 서울의 행정구역('OO구') 하나를 추출해야 합니다.

[추출 규칙]
- 지하철역, 동네 이름, 유명한 장소 등을 바탕으로 정확한 '구' 이름을 찾아주세요.
- 만약 서울이 아니거나, 어느 구인지 특정할 수 없다면 반드시 "알 수 없음"으로 응답해야 합니다.
- 답변은 반드시 'OO구' 또는 '알 수 없음' 형식이어야 합니다.

[예시]
- 사용자 문장: "잠실역 인근" -> 결과: "송파구"
- 사용자 문장: "홍대 쪽" -> 결과: "마포구"
- 사용자 문장: "성수동 카페거리" -> 결과: "성동구"
- 사용자 문장: "판교" -> 결과: "알 수 없음"

[사용자 문장]
"{location}"
"""
    gu_info = await structured_llm.ainvoke(prompt)
    return gu_info.gu_name

async def normalize_disease_name(disease_input: str) -> str:
    """사용자가 입력한 질병명을 표준 질병명으로 정규화합니다."""
    if not disease_input or disease_input.strip().lower() in ['없음', '없어요']:
        return "없음"
    
    # settings.py에 정의된 DISEASE_KEYWORD_MAP을 사용합니다.
    standard_disease_list = list(settings.DISEASE_KEYWORD_MAP.keys())
    prompt_normalize = f"""
당신은 사용자의 문장을 분석하여 가장 관련 있는 표준 질병명을 찾아주는 전문가입니다.
[지침]
1. 아래 [표준 질병 목록]에서 사용자의 [입력 문장]과 가장 관련 있는 항목을 하나만 정확히 골라주세요.
2. 당신의 답변은 반드시 목록에 있는 표준 질병명 단어 그 자체여야 합니다.
3. 어떠한 설명도 추가하지 마세요.
4. 만약 적합한 항목을 찾을 수 없다면, 오직 '해당 없음' 이라고만 답변하세요.

[입력 문장]
{disease_input}

[표준 질병 목록]
{standard_disease_list}
"""
    llm = get_llm()
    normalized_name_result = await llm.ainvoke(prompt_normalize)
    result = normalized_name_result.content.strip().strip("'\" ")
    
    return result if result != '해당 없음' else disease_input
# # services/nlu_service.py
# from langchain_openai import ChatOpenAI
# from models.schemas import UserIntent
# from core.config import settings

# async def extract_user_intent(message: str) -> UserIntent:
#     llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
#     structured_llm = llm.with_structured_output(UserIntent)
#     prompt = f"""당신은 사용자의 답변에서 정보를 추출하여 JSON으로 변환하는 AI입니다.

# [전체 대화 내용]
# {history_string}

# [AI의 마지막 질문]
# {last_bot_question}

# [지침]
# 1. 'AI의 마지막 질문'에 대한 사용자의 답변만을 분석하세요.
# 2. 질문하지 않은 항목이나 답변에 없는 정보는 null로 유지하세요.
# 3. 사용자가 명확하게 '없다', '없음', '없어요' 등으로 부정적으로 대답하면, 반드시 해당 필드를 **문자열 "없음"**으로 채워야 합니다. 이 경우 절대 null로 두지 마세요.

# [예시 1]
# AI의 마지막 질문: 특별히 피해야 할 음식이나 식단(채식 등)이 있으신가요? (없으면 '없음')
# 사용자의 답변: 없음
# 추출 결과에서 dietary_restrictions 필드 값: "없음"

# [예시 2]
# AI의 마지막 질문: 혹시 앓고 계신 질환이 있으신가요? (없으면 '없음')
# 사용자의 답변: 딱히 없어요
# 추출 결과에서 disease 필드 값: "없음"
# """
#     try:
#         return await structured_llm.ainvoke(prompt)
#     except Exception:
#         return UserIntent()


# services/nlu_service.py

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import FullRequestInfo, SeoulGuInfo
from core.config import settings

def get_llm():
    """FastAPI 앱 상태에서 전역 LLM 객체를 가져오는 헬퍼 함수"""
    from main import app
    return app.state.llm

async def extract_info_from_message(conversation_history: List[str], last_bot_question: str) -> FullRequestInfo:
    """
    대화 내용과 AI의 마지막 질문을 바탕으로 사용자 답변에서 정보를 추출합니다.
    (기존 extract_user_intent 함수의 문제를 수정한 버전입니다.)
    """
    structured_llm = get_llm().with_structured_output(FullRequestInfo)
    history_string = "\n".join(conversation_history)

    # ▼▼▼▼▼ 지침을 더 명확하게 수정한 최종 프롬프트 ▼▼▼▼▼
    prompt = f"""당신은 사용자의 최신 답변에서 정보를 추출하는 전문가입니다.

[분석 대상 대화]
{history_string}

[AI의 마지막 질문]
{last_bot_question}

[가장 중요한 규칙]
'AI의 마지막 질문'에 대한 사용자의 답변이 "없음", "없어요", "아니요", "괜찮아요" 등 명백한 부정의 의미라면, 반드시 해당 필드를 **문자열 "없음"**으로 채워야 합니다. 절대 null로 비워두지 마세요.

[추가 지침]
- 'AI의 마지막 질문'과 관련 없는 정보는 추출하지 마세요.
- 답변에 없는 정보는 null로 유지하세요.

[예시]
AI의 마지막 질문: 특별히 피해야 할 음식이 있나요?
사용자 답변: 없음
결과: {{ "dietary_restrictions": "없음" }}
"""
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    try:
        # Pydantic 모델 이름도 FullRequestInfo로 통일합니다.
        return await structured_llm.ainvoke(prompt)
    except Exception as e:
        print(f"[NLU 오류] 정보 추출 실패: {e}")
        return FullRequestInfo()

# --- 이 파일의 다른 함수들 (기존과 동일) ---

async def extract_gu_from_location(location: str) -> str:
    """사용자가 언급한 위치에서 서울시 '구' 이름을 추출합니다."""
    seoul_gu_extractor = get_llm().with_structured_output(SeoulGuInfo)
    prompt = f"사용자 문장: \"{location}\"에서 서울시 행정구 이름을 'OO구' 형식으로 추출해줘. 없으면 '알 수 없음'으로."
    gu_info = await seoul_gu_extractor.ainvoke(prompt)
    return gu_info.gu_name

async def normalize_disease_name(disease_input: str) -> str:
    """사용자가 입력한 질병명을 표준 질병명으로 정규화합니다."""
    if not disease_input or disease_input.strip().lower() in ['없음', '없어요']:
        return "없음"
    
    standard_disease_list = list(settings.DISESE_KEYWORD_MAP.keys())
    prompt_normalize = f"""
당신은 사용자의 문장을 분석하여 가장 관련 있는 표준 질병명을 찾아주는 전문가입니다.
[지침]
1. 아래 [표준 질병 목록]에서 사용자의 [입력 문장]과 가장 관련 있는 항목을 하나만 정확히 골라주세요.
2. 당신의 답변은 반드시 목록에 있는 표준 질병명 단어 그 자체여야 합니다.
3. 어떠한 설명도 추가하지 마세요.
4. 만약 적합한 항목을 찾을 수 없다면, 오직 '해당 없음' 이라고만 답변하세요.

[입력 문장]
{disease_input}

[표준 질병 목록]
{standard_disease_list}
"""
    llm = get_llm()
    normalized_name_result = await llm.ainvoke(prompt_normalize)
    result = normalized_name_result.content.strip().strip("'\" ")
    
    return result if result != '해당 없음' else disease_input