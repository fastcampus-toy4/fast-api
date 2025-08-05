import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Route,
    TimeoutError as PlaywrightTimeoutError,
)
from langchain_openai import ChatOpenAI

from models.schemas import CrawledInfo, FinalRecommendation
from core.config import settings

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PlaywrightRestaurantCrawler:
    """
    Playwright와 LLM을 결합하여 음식점 정보를 효율적으로 크롤링하고 처리하는 클래스.
    """

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
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        await context.route("**/*", self._handle_route)
        return context

    async def _crawl_single_restaurant(self, context: BrowserContext, restaurant: Dict) -> Dict:
        """단일 음식점의 정보를 네이버 지도에서 크롤링합니다."""
        restaurant_full_name = f"{restaurant.get('name', '')} {restaurant.get('branch_name', '')}".strip()
        result_data = {
            **restaurant,
            "restaurant_full_name": restaurant_full_name,
            "crawling_success": False
        }
        page = await context.new_page()
        page.set_default_timeout(self.timeout)

        try:
            await page.goto(
                f"https://map.naver.com/v5/search/{restaurant_full_name}",
                wait_until="domcontentloaded"
            )
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
                    if all_text.strip():
                        break
                except PlaywrightTimeoutError:
                    continue

            if not all_text.strip():
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
        if not restaurants:
            return []

        async with async_playwright() as p:
            browser = await p.chromium.launch(**self.browser_options)
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def worker(r):
                async with semaphore:
                    context = await self._create_browser_context(browser)
                    try:
                        return await self._crawl_single_restaurant(context, r)
                    finally:
                        await context.close()

            tasks = [worker(r) for r in restaurants]
            results = await asyncio.gather(*tasks)
            await browser.close()

        return results


async def _process_with_llm_batch(data_list: List[tuple]) -> List[Optional[Dict]]:
    if not data_list:
        return []
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        prompt = ""
        for i, (raw_text, name) in enumerate(data_list):
            prompt += (
                f'<restaurant index="{i}"><name>{name}</name>'
                f'<crawled_text>{raw_text[:2000]}</crawled_text></restaurant>\n'
            )
        prompt = (
            "다음 여러 음식점의 크롤링된 텍스트에서 정보를 추출하여 JSON으로 반환해주세요.\n"
            "- 각 음식점은 <restaurant> 태그로 구분됩니다.\n"
            "- address, operating_hours, phone_number, holiday_info 필드를 추출합니다.\n"
            "- 정보가 없는 필드는 null로 설정합니다.\n\n"
            f"{prompt}\n"
            "'results' 리스트에 각 결과를 순서대로 포함한 형태로 응답해주세요."
        )
        response = await llm.ainvoke(prompt)
        return json.loads(response.content).get("results", [None] * len(data_list))
    except Exception as e:
        logger.error(f"[LLM 오류] 배치 정형화 실패: {e}")
        return [None] * len(data_list)


async def check_visitable(info: Dict, user_time: str) -> bool:
    """
    단일 음식점에 대해 LLM을 사용해 방문 가능 여부를 판단합니다.
    """
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        prompt = (
            f"음식점 정보: {info}\n"
            f"방문 희망 시간: {user_time}\n"
            "visitable 필드만 true/false로 반환해주세요."
        )
        response = await llm.ainvoke(prompt)
        return json.loads(response.content).get("visitable", False)
    except Exception as e:
        logger.error(f"[LLM 오류] 방문 가능 여부 판단 실패: {e}")
        return False


async def check_visitable_batch(restaurants: List[Dict], user_time: str) -> List[bool]:
    """
    여러 음식점에 대해 방문 가능 여부를 일괄 판단합니다.
    """
    if not restaurants:
        return []
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        weekday = ["월","화","수","목","금","토","일"][datetime.today().weekday()]
        prompt = (
            f"방문 희망 시간: {user_time} ({weekday}요일)\n"
            + "".join(
                f'<restaurant index="{i}"><operating_hours>'
                f"{r.get('operating_hours','정보 없음')}</operating_hours>"
                f"<holiday_info>{r.get('holiday_info','정보 없음')}</holiday_info>"
                f"</restaurant>\n"
                for i, r in enumerate(restaurants)
            )
            + "results 리스트에 true/false 순서대로 반환해주세요."
        )
        response = await llm.ainvoke(prompt)
        return json.loads(response.content).get("results", [False] * len(restaurants))
    except Exception as e:
        logger.error(f"[LLM 오류] 배치 방문 판단 실패: {e}")
        return [False] * len(restaurants)


async def get_final_recommendations_with_crawling(
    final_candidates: List[Dict], user_time: str
) -> List[FinalRecommendation]:
    """
    최종 후보 음식점에 대해 크롤링 + 방문 가능 필터를 적용해
    최종 추천 리스트를 반환합니다.
    """
    if not final_candidates:
        return []

    crawler = PlaywrightRestaurantCrawler(max_concurrent=3)
    crawled = await crawler.crawl_restaurants_batch(final_candidates)

    # 크롤링 성공한 항목만
    success = [r for r in crawled if r.get("crawling_success")]
    open_list: List[FinalRecommendation] = []

    if success:
        visit_flags = await check_visitable_batch(success, user_time)
        for r, flag in zip(success, visit_flags):
            r["is_visitable"] = flag
            if flag:
                open_list.append(FinalRecommendation(**r))
    return open_list
