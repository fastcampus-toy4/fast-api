<<<<<<< HEAD
=======
import json # <--- 이 한 줄을 추가하세요!
>>>>>>> 07a5156 (end)
import asyncio

# services/crawl_service.py
from typing import Dict, List
from langchain_openai import ChatOpenAI
from models.schemas import CrawledInfo # Pydantic 모델 사용
from core.config import settings

<<<<<<< HEAD
# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PlaywrightRestaurantCrawler:
=======

async def get_restaurants_realtime_info(restaurants: List[Dict]) -> List[Dict]:
>>>>>>> 07a5156 (end)
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
<<<<<<< HEAD

    def __init__(self, max_concurrent: int = 3, timeout: int = 30000):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.browser_options = {"headless": True}

    async def _handle_route(self, route: Route):
        """네트워크 요청을 가로채 불필요한 리소스를 차단합니다."""
        if route.request.resource_type in ["image", "font", "stylesheet", "media"]:
            await route.abort()
        else:
            await route.continue_()

    async def _create_browser_context(self, browser: Browser) -> BrowserContext:
        """최적화된 새 브라우저 컨텍스트를 생성합니다."""
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        await context.route("**/*", self._handle_route)
        return context

    async def _process_with_llm_batch(self, data_list: List[tuple]) -> List[Optional[Dict]]:
        """LLM을 사용하여 크롤링된 텍스트 목록에서 정형화된 정보를 일괄 추출합니다."""
        if not data_list:
            return []

        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY, model_kwargs={"response_format": {"type": "json_object"}})

            restaurant_info_prompt = ""
            for i, (raw_text, name) in enumerate(data_list):
                restaurant_info_prompt += f'<restaurant index="{i}"><name>{name}</name><crawled_text>{raw_text[:2000]}</crawled_text></restaurant>\n'

            prompt = f"""다음 여러 음식점의 크롤링된 텍스트에서 정보를 추출하여 JSON으로 반환해주세요.
- 각 음식점은 <restaurant> 태그로 구분됩니다.
- "address", "operating_hours", "phone_number", "holiday_info" 필드를 추출합니다.
- 정보가 없는 필드는 null로 설정합니다.
- '영업중' 같은 실시간 상태는 operating_hours에서 제외해주세요.

{restaurant_info_prompt}

'results' 리스트에 각 음식점 결과를 순서대로 포함하여 아래 JSON 형식으로 응답해주세요.
{{ "results": [ {{ "address": "...", "operating_hours": "...", "phone_number": "...", "holiday_info": "..." }}, ... ] }}
"""
            response = await llm.ainvoke(prompt)
            parsed_data = json.loads(response.content)
            logger.info(f"[LLM] {len(data_list)}건의 정보 정형화 배치 처리 성공")
            return parsed_data.get("results", [None] * len(data_list))

        except Exception as e:
            logger.error(f"[LLM 오류] 정보 정형화 배치 처리 실패: {e}")
            return [None] * len(data_list)

    async def _crawl_single_restaurant(self, context: BrowserContext, restaurant: Dict) -> Dict:
        """단일 음식점의 정보를 네이버 지도에서 크롤링합니다."""
        restaurant_full_name = f"{restaurant.get('name', '')} {restaurant.get('branch_name', '')}".strip()
        result_data = {**restaurant, "restaurant_full_name": restaurant_full_name, "crawling_success": False}
        page = await context.new_page()
        page.set_default_timeout(self.timeout)

        try:
            await page.goto(f"https://map.naver.com/v5/search/{restaurant_full_name}", wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            search_iframe = page.frame_locator("#searchIframe")
            entry_iframe = page.frame_locator("#entryIframe")

            try:
                await search_iframe.locator("a.place_bluelink").first.click(timeout=10000)
                await page.wait_for_timeout(2000)
            except PlaywrightTimeoutError:
                logger.info(f"'{restaurant_full_name}'은(는) 상세 페이지로 바로 연결됩니다.")

            info_selectors = ["div.place_section", "div.PIbes", "body"]
            all_text = ""
            for selector in info_selectors:
                try:
                    all_text = await entry_iframe.locator(selector).first.inner_text(timeout=5000)
                    if all_text and all_text.strip(): break
                except PlaywrightTimeoutError: continue

            if not all_text or not all_text.strip():
                raise ValueError("크롤링된 텍스트가 비어 있습니다.")

            result_data["raw_crawling_data"] = all_text.strip()
            result_data["crawling_success"] = True
            logger.info(f"[크롤링 성공] '{restaurant_full_name}'")

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            result_data["crawling_error"] = error_message
            logger.error(f"[크롤링 실패] '{restaurant_full_name}': {error_message}")
        finally:
            await page.close()
        return result_data

    async def crawl_restaurants_batch(self, restaurants: List[Dict]) -> List[Dict]:
        """주어진 음식점 목록을 비동기 및 배치 처리 방식으로 크롤링합니다."""
        if not restaurants: return []

        async with async_playwright() as p:
            browser = await p.chromium.launch(**self.browser_options)
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def crawl_with_semaphore(restaurant):
                async with semaphore:
                    context = await self._create_browser_context(browser)
                    try:
                        return await self._crawl_single_restaurant(context, restaurant)
                    finally:
                        await context.close()

            tasks = [crawl_with_semaphore(r) for r in restaurants]
            crawled_results = await asyncio.gather(*tasks)
            await browser.close()

        llm_processing_data = []
        llm_target_indices = []
        for i, result in enumerate(crawled_results):
            if result.get("crawling_success"):
                llm_processing_data.append((result["raw_crawling_data"], result["restaurant_full_name"]))
                llm_target_indices.append(i)

        if llm_processing_data:
            llm_results = await self._process_with_llm_batch(llm_processing_data)
            for i, llm_result in enumerate(llm_results):
                if llm_result:
                    original_index = llm_target_indices[i]
                    crawled_results[original_index].update(llm_result)

        return crawled_results
=======
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
>>>>>>> 07a5156 (end)


async def check_visitable(info: Dict, user_time: str) -> bool:
    """
    크롤링된 영업시간 정보와 사용자의 방문 희망 시간을 바탕으로
    LLM을 이용해 현재 방문 가능한지 여부를 판단합니다.
    """
<<<<<<< HEAD
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
=======
    # model_kwargs를 사용하여 JSON 모드 활성화
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0, 
>>>>>>> 07a5156 (end)
        api_key=settings.OPENAI_API_KEY,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )
<<<<<<< HEAD

=======
    
>>>>>>> 07a5156 (end)
    prompt = f"""
    아래 음식점 정보와 사용자 방문 희망 시간을 보고, 현재 방문이 가능한지 판단해주세요.
    "visitable" 필드에 true 또는 false 값만 포함하는 JSON 형식으로만 답변해주세요.
    어떠한 설명도 추가하지 말고 JSON 객체만 응답해야 합니다.

<<<<<<< HEAD
    음식점 정보: {info}
    사용자 희망 방문 시간: {user_time}
    """

    try:
        response = await llm.ainvoke(prompt)
        parsed = json.loads(response.content)
        return parsed.get("visitable", False)
    except Exception as e:
        logger.error(f"[LLM 오류] 방문 가능 여부 판단 실패: {e}")
        return False


async def check_visitable_batch(restaurants: List[Dict], user_time: str) -> List[bool]:
    """LLM을 사용하여 여러 음식점의 방문 가능 여부를 일괄 확인합니다."""
    if not restaurants: return []
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY, model_kwargs={"response_format": {"type": "json_object"}})
        today_weekday = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][datetime.today().weekday()]

        restaurant_info_prompt = ""
        for i, r in enumerate(restaurants):
            restaurant_info_prompt += f'<restaurant index="{i}"><operating_hours>{r.get("operating_hours", "정보 없음")}</operating_hours><holiday_info>{r.get("holiday_info", "정보 없음")}</holiday_info></restaurant>\n'

        prompt = f"""방문 희망 시간은 '{user_time} ({today_weekday})'입니다.
아래 각 음식점의 영업 정보를 보고, 방문 가능한지 판단해주세요.

{restaurant_info_prompt}

'results' 리스트에 각 음식점의 방문 가능 여부를 true 또는 false로 순서대로 포함하여 JSON으로 응답해주세요.
{{ "results": [true/false, ...] }}
"""
        response = await llm.ainvoke(prompt)
        parsed_data = json.loads(response.content)
        logger.info(f"[LLM] {len(restaurants)}건의 방문 가능 여부 배치 처리 성공")
        return parsed_data.get("results", [False] * len(restaurants))
    except Exception as e:
        logger.error(f"[LLM 오류] 방문 가능 여부 배치 처리 실패: {e}")
        return [False] * len(restaurants)


async def get_final_recommendations_with_crawling(
    final_candidates: List[Dict], user_time: str
) -> List[FinalRecommendation]:
    """
    최종 후보 음식점 목록에 대해 실시간 크롤링 및 LLM 기반 영업 여부 필터링을 수행합니다.
    """
    logger.info(f"--- 실시간 정보 확인 및 최종 필터링 시작 ({len(final_candidates)}곳 대상) ---")
    if not final_candidates:
        return []

    crawler = PlaywrightRestaurantCrawler(max_concurrent=3)
    crawled_restaurants = await crawler.crawl_restaurants_batch(final_candidates)

    successful_crawls = [r for r in crawled_restaurants if r.get("crawling_success")]
    open_restaurants = []

    if successful_crawls:
        visitable_results = await check_visitable_batch(successful_crawls, user_time)

        for i, restaurant in enumerate(successful_crawls):
            if i < len(visitable_results) and visitable_results[i]:
                restaurant["is_visitable"] = True
                open_restaurants.append(FinalRecommendation(**restaurant))
                logger.info(f"[방문 가능] '{restaurant['restaurant_full_name']}'")
            else:
                restaurant["is_visitable"] = False
                logger.info(f"[방문 불가] '{restaurant['restaurant_full_name']}'")

    logger.info(f"--- 최종 추천: {len(open_restaurants)}곳 ---")
    return open_restaurants
=======
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
>>>>>>> 07a5156 (end)
