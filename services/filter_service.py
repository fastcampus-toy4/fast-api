# import asyncio
# import json
# import pandas as pd
# from typing import Set, List, Dict, Optional
# import re

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.exceptions import OutputParserException
# from langchain_google_community import GoogleSearchAPIWrapper
# from sqlalchemy import text
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.orm import sessionmaker

# from . import data_loader  # 사전 로드된 DB 및 데이터 사용
# from core.config import settings

# # --- 비동기 DB 세션 생성 헬퍼 ---
# # chat_orchestrator에서 세션을 넘겨주지 않으므로, 필요할 때 직접 생성합니다.
# async def get_db_session() -> AsyncSession:
#     # 비동기 DB 엔진 생성 (설정은 config.py에서 가져옴)
#     async_db_uri = (
#         f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
#         f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB_NAME}"
#     )
#     engine = create_async_engine(async_db_uri)
    
#     # 비동기 세션 메이커 생성
#     AsyncSessionLocal = sessionmaker(
#         bind=engine, class_=AsyncSession, expire_on_commit=False
#     )
#     async with AsyncSessionLocal() as session:
#         yield session

# # --- 1. 건강 정보 기반 메뉴 필터링 (고급 버전) ---

# async def filter_menus_by_health(standard_dishes: Set[str], disease: str, dietary_restrictions: str) -> Set[str]:
#     """
#     [고급] 동적 수치 필터링과 RAG 자가 교정을 결합한 메뉴 필터링 함수.
#     """
#     print("\n--- 3단계: 건강 정보 기반 메뉴 필터링 (고급) 시작 ---")
#     print(f"대상 메뉴: {len(standard_dishes)}개, 질병: {disease}")

#     if not disease or disease.strip().lower() in ['없음', '없어요']:
#         print("-> 질병 정보 없음. 모든 메뉴를 적합으로 간주합니다.")
#         return standard_dishes

#     # --- [1단계] 동적 수치 기반 사전 필터링 (MySQL 사용) ---
#     pre_filtered_dishes = set()
#     try:
#         # 로컬 DB에서 영양 기준 정보 검색
#         retrieved_standards = await _get_nutrition_standards_from_local_db(disease)
#         if retrieved_standards:
#             # 영양 기준에서 수치 조건 추출 (예: {"max_sodium_mg": 500})
#             criteria = await _extract_criteria_from_text(retrieved_standards, disease)
#             if criteria:
#                 # 추출된 조건으로 MySQL에서 적합 음식 검색
#                 async for db in get_db_session():
#                     suitable_foods_df = await _search_foods_in_mysql(db, criteria)
#                     if not suitable_foods_df.empty:
#                         mysql_food_names = set(suitable_foods_df['name'].str.strip().lower())
#                         # 현재 메뉴 목록과 MySQL 결과를 비교하여 필터링
#                         for dish in standard_dishes:
#                             if dish.strip().lower() in mysql_food_names:
#                                 pre_filtered_dishes.add(dish)
                        
#                         if pre_filtered_dishes:
#                             print(f"-> 동적 수치 필터링 통과: {len(pre_filtered_dishes)}개 메뉴")
#                             standard_dishes = pre_filtered_dishes # LLM 검증 대상을 줄임
#     except Exception as e:
#         print(f"[동적 수치 필터링 오류] {e}. LLM 검증을 계속 진행합니다.")


#     # --- [2단계] LLM 기반 RAG 필터링 (자가 교정 포함) ---
#     llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    
#     # 정보 검색 (ChromaDB 또는 웹)
#     retrieved_knowledge = await _retrieve_health_knowledge(disease, dietary_restrictions)

#     # 1차 LLM 판단
#     initial_judgment = await _get_llm_judgment(llm, retrieved_knowledge, standard_dishes, is_final=False)
    
#     # 1차 통과 메뉴 선별
#     suitable_after_initial = {
#         dish for dish, details in initial_judgment.items()
#         if isinstance(details, dict) and details.get("is_suitable")
#     }

#     if not suitable_after_initial:
#         print("-> 1차 LLM 필터링에서 적합한 메뉴를 찾지 못했습니다.")
#         return set()

#     print(f"-> 1차 LLM 필터링 통과 메뉴: {list(suitable_after_initial)}")

#     # 2차 LLM 재검증 (자가 교정)
#     final_judgment_result = await _get_llm_judgment(llm, retrieved_knowledge, suitable_after_initial, is_final=True, initial_judgment=initial_judgment)
    
#     final_suitable_dishes = {
#         dish for dish, details in final_judgment_result.items()
#         if isinstance(details, dict) and details.get("is_suitable")
#     }
    
#     print(f"✅ 최종 필터링 완료. {len(final_suitable_dishes)}개 메뉴 통과.")
#     return final_suitable_dishes


# # --- 2. 리뷰 기반 음식점 필터링 ---

# async def filter_restaurants_by_review(restaurants: List[Dict], other_requests: str) -> List[Dict]:
#     """리뷰 기반으로 음식점 목록을 필터링하고 재정렬"""
#     if not other_requests or other_requests.lower() in ['없음', '없어요']:
#         return restaurants

#     def search_reviews_sync():
#         return data_loader.REVIEW_DB.similarity_search_with_score(other_requests, k=30)
    
#     retrieved_reviews = await asyncio.to_thread(search_reviews_sync)

#     if not retrieved_reviews:
#         return restaurants

#     candidate_names = {f"{r['name']} {r.get('branch_name', '')}".strip() for r in restaurants}
#     scores = {name: 0.0 for name in candidate_names}
#     counts = {name: 0 for name in candidate_names}

#     for doc, score in retrieved_reviews:
#         name = doc.metadata.get("restaurant_full_name")
#         if name in scores:
#             scores[name] += score
#             counts[name] += 1
    
#     # 리뷰가 있는 식당만 필터링하고, (총점/리뷰 수)로 평균 점수 계산 (낮을수록 좋음)
#     ranked_names = sorted(
#         [name for name, count in counts.items() if count > 0],
#         key=lambda name: scores[name] / counts[name]
#     )
    
#     restaurants_dict = {f"{r['name']} {r.get('branch_name', '')}".strip(): r for r in restaurants}
#     final_list = [restaurants_dict[name] for name in ranked_names if name in restaurants_dict]

#     return final_list


# # --- 헬퍼 함수들 ---

# async def _get_nutrition_standards_from_local_db(disease: str) -> str:
#     """로컬 ChromaDB에서 질병의 영양 기준 정보 검색"""
#     query = f"{disease} 환자의 식단 가이드라인, 영양소별 권장 또는 제한 섭취량 (mg, g, kcal 등 수치 정보 위주)"
    
#     def search_db_sync():
#         docs = data_loader.HEALTH_JUDGMENT_DB.similarity_search(query, k=3)
#         return "\n\n".join(doc.page_content for doc in docs) if docs else ""
        
#     return await asyncio.to_thread(search_db_sync)

# async def _extract_criteria_from_text(text: str, disease: str) -> dict:
#     """LLM을 사용해 텍스트에서 영양 조건(JSON) 추출"""
#     prompt = PromptTemplate.from_template(
#         """아래 {disease} 환자의 영양 기준 정보에서, MySQL 컬럼과 매칭될 수 있는 수치 조건을 JSON으로 추출해줘.
#         - 나트륨(sodium_mg), 콜레스테롤(cholesterol_mg), 단백질(protein_g), 지방(fat_g), 칼로리(kcal) 등
#         - '이하'는 'max_', '이상'은 'min_'을 사용. 예: {{"max_sodium_mg": 2000, "min_protein_g": 20}}
#         - 정보에 언급된 것만 포함하고, 없으면 빈 JSON {{}} 반환.
        
#         [영양 기준 정보]
#         {context}
        
#         JSON:
#         """
#     )
#     llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
#     chain = prompt | llm | JsonOutputParser()
#     try:
#         return await chain.ainvoke({"disease": disease, "context": text})
#     except Exception:
#         return {}

# async def _search_foods_in_mysql(db: AsyncSession, criteria: dict) -> pd.DataFrame:
#     """동적 조건으로 MySQL에서 적합 음식 검색"""
#     conditions, params = [], {}
#     for key, value in criteria.items():
#         op = "<=" if key.startswith("max_") else ">="
#         column = key.replace("max_", "").replace("min_", "")
#         conditions.append(f"{column} {op} :{key}")
#         params[key] = value

#     if not conditions: return pd.DataFrame()

#     query_str = f"SELECT name FROM food_nutrition WHERE {' AND '.join(conditions)} LIMIT 200"
#     result = await db.execute(text(query_str), params)
#     return pd.DataFrame(result.fetchall(), columns=['name'])

# async def _retrieve_health_knowledge(disease: str, dietary_restrictions: str) -> str:
#     """ChromaDB 또는 웹에서 일반 건강 정보 검색"""
#     query = f"{disease} 환자에게 추천하는 {dietary_restrictions or ''} 식단"
    
#     def search_db_sync():
#         docs = data_loader.HEALTH_JUDGMENT_DB.similarity_search(query, k=2)
#         return "\n\n".join(doc.page_content for doc in docs) if docs else ""

#     knowledge = await asyncio.to_thread(search_db_sync)

#     if not knowledge or len(knowledge) < 100:
#         try:
#             print("-> 로컬 DB 정보 부족, 웹 검색 실행...")
#             search_tool = GoogleSearchAPIWrapper()
#             web_results = await asyncio.to_thread(search_tool.run, f"{query} 최신 가이드라인")
#             if web_results and "No good Google Search Result" not in web_results:
#                 llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
#                 summary = await llm.ainvoke(f"다음 웹 검색 결과를 바탕으로 '{query}'에 대한 핵심만 요약해줘:\n\n{web_results}")
#                 knowledge += "\n\n[웹 검색 정보]\n" + summary.content
#         except Exception as e:
#             print(f"웹 검색 오류: {e}")

#     return knowledge if knowledge.strip() else "관련 건강 정보를 찾지 못했습니다."

# async def _get_llm_judgment(llm: ChatOpenAI, knowledge: str, dishes: Set[str], is_final: bool, initial_judgment: Optional[Dict] = None) -> Dict:
#     """LLM을 통해 메뉴 적합성 판단 (1차/최종)"""
#     if is_final:
#         template = """당신은 엄격한 임상 영양사입니다. 아래 1차 판단 결과를 비판적으로 재검증해주세요.
#         [건강 정보]: {knowledge}
#         [1차 판단 대상]: {dishes}
        
#         [재검증 규칙]
#         - 질병 악화 가능성이 조금이라도 있거나, 애매한 음식은 반드시 '부적합' 처리하세요.
#         - 안전이 최우선입니다. 확실히 도움 되는 것만 '적합'으로 최종 판단합니다.
        
#         JSON 형식으로만 최종 판단 결과를 응답해주세요:
#         """
#     else:
#         template = """당신은 임상 영양사입니다. 아래 건강 정보를 바탕으로 각 음식이 질병에 적합한지 판단해주세요.
#         [건강 정보]: {knowledge}
#         [음식 목록]: {dishes}
        
#         [규칙]
#         - 건강 정보에 기반하여 도움이 되거나, 최소한 해롭지 않은 음식만 '적합'으로 판단하세요.
#         - 결과는 반드시 JSON 형식으로만 응답해야 합니다.
#         """

#     prompt = ChatPromptTemplate.from_template(template)
#     chain = prompt | llm | JsonOutputParser()
    
#     try:
#         return await chain.ainvoke({
#             "knowledge": knowledge,
#             "dishes": json.dumps(list(dishes), ensure_ascii=False)
#         })
#     except OutputParserException as e:
#         print(f"[LLM {'최종' if is_final else '1차'} 판단 오류] {e}")
#         # 오류 시, 1차 판단의 경우 모두 통과, 최종 판단의 경우 1차 통과 메뉴를 그대로 반환
#         return {dish: {"is_suitable": True, "reason": "오류로 인한 통과"} for dish in dishes}

# services/filter_service.py

import asyncio
import json
import pandas as pd
from typing import Set, List, Dict, Optional
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_community import GoogleSearchAPIWrapper
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from . import data_loader  # 사전 로드된 DB 및 데이터 사용
from core.config import settings

# --- 비동기 DB 세션 생성 헬퍼 ---
async def get_db_session() -> AsyncSession:
    async_db_uri = (
        f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB_NAME}"
    )
    engine = create_async_engine(async_db_uri, pool_recycle=3600)
    AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionLocal() as session:
        yield session

# --- [메인 함수] 건강 정보 필터링 전체 프로세스 ---
async def filter_menus_by_health(standard_dishes: Set[str], disease: str, dietary_restrictions: str) -> Set[str]:
    """
    [최적화] 1차 사전판단 필터링으로 후보군을 줄이고, 2단계 심층 필터링으로 최종 검증합니다.
    """
    if not disease or disease.lower() in ['없음', '없어요']:
        return standard_dishes

    print("\n--- [1단계] 사전 판단 데이터 기반 필터링 시작 ---")
    pre_filtered_dishes = await _initial_filter_with_prejudgments(standard_dishes, disease)

    if not pre_filtered_dishes:
        print("-> 1차 필터링에서 적합한 메뉴 후보를 찾지 못했습니다.")
        return set()

    print(f"-> 1차 필터링 통과: {len(pre_filtered_dishes)}개 메뉴 후보 선별")

    # 압축된 후보군을 대상으로 심층 필터링(하이브리드 + 자가 교정) 수행
    async for db in get_db_session():
        final_dishes = await _hybrid_deep_filter(db, pre_filtered_dishes, disease, dietary_restrictions)
        print(f"✅ 최종 필터링 완료. {len(final_dishes)}개 메뉴 통과.")
        return final_dishes

# --- [헬퍼 함수 1] 1차 사전판단 필터링 ---
async def _initial_filter_with_prejudgments(standard_dishes: Set[str], disease: str) -> Set[str]:
    """미리 계산된 health_judgments DB를 사용해 1차적으로 메뉴를 필터링합니다."""
    suitable_dishes = set()
    food_to_cluster = data_loader.FOOD_TO_CLUSTER_MAP
    cluster_to_repre = data_loader.CLUSTER_TO_FOOD_MAP

    async def check_suitability(dish: str):
        cluster_id = food_to_cluster.get(dish)
        if cluster_id is None: return None

        representative_food = cluster_to_repre.get(cluster_id)
        if representative_food is None: return None

        def search_judgment_sync():
            try:
                results = data_loader.HEALTH_JUDGMENT_DB.get(
                    where={"$and": [{"disease": {"$eq": disease}}, {"food_name": {"$eq": representative_food}}]},
                    limit=1
                )
                if results and results['metadatas'] and results['metadatas'][0].get("is_suitable"):
                    return True
            except Exception as e:
                print(f"[1차 필터링 DB 조회 오류] {e}")
            return False

        if await asyncio.to_thread(search_judgment_sync):
            return dish
        return None

    tasks = [check_suitability(dish) for dish in standard_dishes]
    results = await asyncio.gather(*tasks)
    return {dish for dish in results if dish is not None}

# --- [헬퍼 함수 2] 2차 심층 필터링 (고급 버전) ---
async def _hybrid_deep_filter(db: AsyncSession, standard_dishes: Set[str], disease: str, dietary_restrictions: str) -> Set[str]:
    """[고급] 동적 수치 필터링과 RAG 자가 교정을 결합한 메뉴 필터링 함수."""
    print("\n--- [2단계] 하이브리드 심층 필터링 시작 ---")

    # [2-1단계] 동적 수치 기반 사전 필터링 (MySQL 사용)
    try:
        retrieved_standards = await _get_nutrition_standards_from_local_db(disease)
        if retrieved_standards:
            criteria = await _extract_criteria_from_text(retrieved_standards, disease)
            if criteria:
                suitable_foods_df = await _search_foods_in_mysql(db, criteria)
                if not suitable_foods_df.empty:
                    mysql_food_names = set(suitable_foods_df['name'].str.strip().lower())
                    pre_filtered_dishes = {dish for dish in standard_dishes if dish.strip().lower() in mysql_food_names}
                    if pre_filtered_dishes:
                        print(f"-> 동적 수치 필터링 통과: {len(pre_filtered_dishes)}개 메뉴")
                        standard_dishes = pre_filtered_dishes
    except Exception as e:
        print(f"[동적 수치 필터링 오류] {e}")

    # [2-2단계] LLM 기반 RAG 필터링 (자가 교정 포함)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    retrieved_knowledge = await _retrieve_health_knowledge(disease, dietary_restrictions)
    initial_judgment = await _get_llm_judgment(llm, retrieved_knowledge, standard_dishes, is_final=False)
    suitable_after_initial = {dish for dish, details in initial_judgment.items() if isinstance(details, dict) and details.get("is_suitable")}

    if not suitable_after_initial:
        print("-> 2단계 심층 필터링에서 적합한 메뉴를 찾지 못했습니다.")
        return set()

    final_judgment_result = await _get_llm_judgment(llm, retrieved_knowledge, suitable_after_initial, is_final=True)
    return {dish for dish, details in final_judgment_result.items() if isinstance(details, dict) and details.get("is_suitable")}

# --- 리뷰 기반 음식점 필터링 ---
async def filter_restaurants_by_review(restaurants: List[Dict], other_requests: str) -> List[Dict]:
    """리뷰 기반으로 음식점 목록을 필터링하고 재정렬"""
    if not other_requests or other_requests.lower() in ['없음', '없어요']:
        return restaurants

    def search_reviews_sync():
        return data_loader.REVIEW_DB.similarity_search_with_score(other_requests, k=30)

    retrieved_reviews = await asyncio.to_thread(search_reviews_sync)

    if not retrieved_reviews:
        return restaurants

    candidate_names = {f"{r['name']} {r.get('branch_name', '')}".strip() for r in restaurants}
    scores = {name: 0.0 for name in candidate_names}
    counts = {name: 0 for name in candidate_names}

    for doc, score in retrieved_reviews:
        name = doc.metadata.get("restaurant_full_name")
        if name in scores:
            scores[name] += score
            counts[name] += 1

    ranked_names = sorted(
        [name for name, count in counts.items() if count > 0],
        key=lambda name: scores[name] / counts[name]
    )

    restaurants_dict = {f"{r['name']} {r.get('branch_name', '')}".strip(): r for r in restaurants}
    final_list = [restaurants_dict[name] for name in ranked_names if name in restaurants_dict]

    return final_list

# --- 2차 심층 필터링을 위한 헬퍼 함수들 ---
async def _get_nutrition_standards_from_local_db(disease: str) -> str:
    """로컬 ChromaDB에서 질병의 영양 기준 정보 검색"""
    query = f"{disease} 환자의 식단 가이드라인, 영양소별 권장 또는 제한 섭취량 (mg, g, kcal 등 수치 정보 위주)"

    def search_db_sync():
        # health_judgments 대신 food DB를 사용하도록 수정 (사용자 요청 반영)
        docs = data_loader.CHROMA_DB_FOOD.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in docs) if docs else ""

    return await asyncio.to_thread(search_db_sync)

async def _extract_criteria_from_text(text: str, disease: str) -> dict:
    """LLM을 사용해 텍스트에서 영양 조건(JSON) 추출"""
    prompt = PromptTemplate.from_template(
        """아래 {disease} 환자의 영양 기준 정보에서, MySQL 컬럼과 매칭될 수 있는 수치 조건을 JSON으로 추출해줘.
        - 나트륨(sodium_mg), 콜레스테롤(cholesterol_mg), 단백질(protein_g), 지방(fat_g), 칼로리(kcal), 탄수화물(carb_g) 등
        - '이하'는 'max_', '이상'은 'min_'을 사용. 예: {{"max_sodium_mg": 2000, "min_protein_g": 20}}
        - 정보에 언급된 것만 포함하고, 없으면 빈 JSON {{}} 반환.
        
        [영양 기준 정보]
        {context}
        
        JSON:
        """
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    chain = prompt | llm | JsonOutputParser()
    try:
        return await chain.ainvoke({"disease": disease, "context": text})
    except Exception as e:
        print(f"[영양 조건 추출 오류] {e}")
        return {}

async def _search_foods_in_mysql(db: AsyncSession, criteria: dict) -> pd.DataFrame:
    """동적 조건으로 MySQL에서 적합 음식 검색"""
    conditions, params = [], {}
    for key, value in criteria.items():
        op = "<=" if key.startswith("max_") else ">="
        column = key.replace("max_", "").replace("min_", "")
        conditions.append(f"{column} {op} :{key}")
        params[key] = value

    if not conditions: return pd.DataFrame()

    query_str = f"SELECT name FROM food_nutrition WHERE {' AND '.join(conditions)} LIMIT 200"
    result = await db.execute(text(query_str), params)
    return pd.DataFrame(result.fetchall(), columns=['name'])

async def _retrieve_health_knowledge(disease: str, dietary_restrictions: str) -> str:
    """ChromaDB 또는 웹에서 일반 건강 정보 검색"""
    query = f"{disease} 환자에게 추천하는 {dietary_restrictions or ''} 식단"

    def search_db_sync():
        docs = data_loader.CHROMA_DB_FOOD.similarity_search(query, k=2)
        return "\n\n".join(doc.page_content for doc in docs) if docs else ""

    knowledge = await asyncio.to_thread(search_db_sync)

    if not knowledge or len(knowledge) < 100:
        try:
            print("-> 로컬 DB 정보 부족, 웹 검색 실행...")
            search_tool = GoogleSearchAPIWrapper(api_key=settings.GOOGLE_API_KEY, cse_id=settings.GOOGLE_CSE_ID)
            web_results = await asyncio.to_thread(search_tool.run, f"{query} 최신 가이드라인")
            if web_results and "No good Google Search Result" not in web_results:
                llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
                summary = await llm.ainvoke(f"다음 웹 검색 결과를 바탕으로 '{query}'에 대한 핵심만 요약해줘:\n\n{web_results}")
                knowledge += "\n\n[웹 검색 정보]\n" + summary.content
        except Exception as e:
            print(f"웹 검색 오류: {e}")

    return knowledge if knowledge.strip() else "관련 건강 정보를 찾지 못했습니다."

async def _get_llm_judgment(llm: ChatOpenAI, knowledge: str, dishes: Set[str], is_final: bool) -> Dict:
    """LLM을 통해 메뉴 적합성 판단 (1차/최종)"""
    if is_final:
        template = """당신은 임상 영양사입니다. 아래 1차 판단 결과를 비판적으로 재검증해주세요.
        [건강 정보]: {knowledge}
        [1차 판단 대상]: {dishes}
        
        [재검증 규칙]
        - 질병 악화 가능성이 조금이라도 있거나, 애매한 음식은 '부적합' 처리하세요.
        
        JSON 형식으로만 최종 판단 결과를 응답해주세요:
        {{"음식명1": {{"is_suitable": boolean, "reason": "재검증 후 최종 이유"}}, ...}}
        """
    else:
        template = """당신은 임상 영양사입니다. 아래 건강 정보를 바탕으로 각 음식이 질병에 적합한지 판단해주세요.
        [건강 정보]: {knowledge}
        [음식 목록]: {dishes}
        
        [규칙]
        - 건강 정보에 기반하여 도움이 되거나, 최소한 해롭지 않은 음식만 '적합'으로 판단하세요.
        - 결과는 반드시 JSON 형식으로만 응답해야 합니다:
        {{"음식명1": {{"is_suitable": boolean, "reason": "이유"}}, ...}}
        """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | JsonOutputParser()
    
    try:
        return await chain.ainvoke({
            "knowledge": knowledge,
            "dishes": json.dumps(list(dishes), ensure_ascii=False)
        })
    except OutputParserException as e:
        print(f"[LLM {'최종' if is_final else '1차'} 판단 오류] {e}")
        return {dish: {"is_suitable": True, "reason": "오류로 인한 통과"} for dish in dishes}
