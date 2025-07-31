# services/filter_service.py
import asyncio
import json
from typing import Set, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from . import data_loader # 사전 로드된 DB 및 데이터 사용
from core.config import settings

async def filter_menus_by_health(standard_dishes: Set[str], disease: str, dietary_restrictions: str) -> Set[str]:
    """RAG와 LLM을 사용하여 건강/식단 제약 기반으로 메뉴를 필터링 (비동기 처리)"""
    if not disease or disease.lower() in ['없음', '없어요']:
        return standard_dishes

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)

    # 1. 정보 검색 (Retrieve) - ChromaDB 검색은 동기 함수이므로 to_thread로 실행
    def search_db():
        query = f"{disease} 환자에게 추천하는 {dietary_restrictions or ''} 식단"
        docs = data_loader.HEALTH_JUDGMENT_DB.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs]) if docs else "관련 건강 정보를 찾지 못했습니다."

    retrieved_knowledge = await asyncio.to_thread(search_db)

    # 2. LLM 추론 (Generate) - 비동기 호출 (ainvoke)
    prompt_template = """
    [건강 정보]: {retrieved_knowledge}
    [음식 목록]: {dishes}
    
    위 [건강 정보]를 바탕으로, 각 음식이 사용자의 질병({disease})과 식단 제약({dietary_restrictions})에
    적합한지 판단해주세요. 결과는 반드시 다음 JSON 형식으로만 응답해야 합니다.
    {{"음식명1": {{"is_suitable": boolean, "reason": "이유"}}, "음식명2": ...}}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | JsonOutputParser()

    try:
        result = await chain.ainvoke({
            "retrieved_knowledge": retrieved_knowledge,
            "dishes": json.dumps(list(standard_dishes), ensure_ascii=False),
            "disease": disease,
            "dietary_restrictions": dietary_restrictions or "없음",
        })
        
        suitable_dishes = {dish for dish, details in result.items() if details.get("is_suitable")}
        return suitable_dishes
    except Exception as e:
        print(f"[LLM 필터링 오류] {e}")
        return standard_dishes # 오류 발생 시 모든 메뉴를 통과시켜 다음 단계로

async def filter_restaurants_by_review(restaurants: List[Dict], other_requests: str) -> List[Dict]:
    """리뷰 기반으로 음식점 목록을 필터링하고 재정렬"""
    if not other_requests or other_requests.lower() in ['없음', '없어요']:
        return restaurants

    # ChromaDB 검색은 동기 함수이므로 to_thread로 감싸 비동기 컨텍스트에서 안전하게 실행
    def search_reviews():
        return data_loader.REVIEW_DB.similarity_search_with_score(other_requests, k=30)
    
    retrieved_reviews = await asyncio.to_thread(search_reviews)

    if not retrieved_reviews:
        return restaurants

    # 점수 계산 (이 부분은 CPU bound이므로 그냥 둬도 무방)
    scores = {f"{r['name']} {r.get('branch_name', '')}".strip(): 0.0 for r in restaurants}
    for doc, score in retrieved_reviews:
        name = doc.metadata.get("restaurant_full_name")
        if name in scores:
            scores[name] += score  # score가 거리(distance)이므로 낮을수록 좋음

    # 점수가 낮은 순(유사도가 높은 순)으로 정렬
    ranked_names = sorted(scores.keys(), key=lambda name: scores[name])
    
    # 정렬된 이름 순서대로 최종 음식점 목록 생성
    restaurants_dict = {f"{r['name']} {r.get('branch_name', '')}".strip(): r for r in restaurants}
    final_list = [restaurants_dict[name] for name in ranked_names if name in restaurants_dict]

    return final_list