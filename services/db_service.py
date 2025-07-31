# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import text
# from typing import List, Dict, Set, Optional

# async def get_restaurants_from_db(db: AsyncSession, location_gu: str, amenities: List[str]) -> List[Dict]:
#     """
#     위치(구)와 편의시설 조건으로 DB에서 1차 후보 음식점 목록을 조회합니다.
#     SQLAlchemy의 매개변수 바인딩을 사용하여 SQL 인젝션을 안전하게 방지합니다.
#     """
#     where_clauses = ["region_name LIKE :location_gu"]
#     params = {"location_gu": f"%{location_gu}%"}

#     amenity_map = {"주차": "has_parking", "와이파이": "has_wifi", "놀이방": "has_kids_zone"}
#     for i, amenity in enumerate(amenities):
#         if amenity in amenity_map:
#             param_name = f"amenity_{i}"
#             where_clauses.append(f"{amenity_map[amenity]} = :{param_name}")
#             params[param_name] = 'Y'

#     query_str = f"SELECT name, branch_name FROM restaurant WHERE {' AND '.join(where_clauses)} LIMIT 100"
#     query = text(query_str)
    
#     result = await db.execute(query, params)
#     return [dict(row) for row in result.mappings()]

# async def get_normalized_menus_for_restaurants(db: AsyncSession, restaurants: List[Dict]) -> Dict[str, Set[str]]:
#     """선별된 음식점 목록에 대해 정규화된 메뉴를 조회합니다."""
#     if not restaurants:
#         return {}

#     restaurant_names = list(set(r['name'] for r in restaurants))
#     params = {"names": restaurant_names}
    
#     # ===================================================================
#     # [수정] MySQL 문법에 맞게 '= ANY(:names)'를 'IN :names'로 변경합니다.
#     # SQLAlchemy가 :names 파라미터를 (value1, value2, ...) 형태로 자동 변환해줍니다.
#     # ===================================================================
#     query = text("""
#         SELECT restaurant_name, branch_name, normalized_name 
#         FROM restaurant_menu 
#         WHERE restaurant_name IN :names AND normalized_name IS NOT NULL
#     """)
#     # ===================================================================

#     result = await db.execute(query, params)
    
#     menus_by_restaurant = {}
#     for row in result.mappings():
#         # 지점명이 없는 경우를 안전하게 처리 (branch_name or '')
#         full_name = f"{row['restaurant_name']} {row.get('branch_name', '') or ''}".strip()
#         if full_name not in menus_by_restaurant:
#             menus_by_restaurant[full_name] = set()
#         menus_by_restaurant[full_name].add(row['normalized_name'])
        
#     return menus_by_restaurant

# async def log_interaction(
#     db: AsyncSession,
#     session_id: str,
#     step_name: str,
#     log_type: str,
#     content: dict | str,
#     user_id: Optional[str] = None
# ):
#     """(비동기) 사용자 상호작용 로그를 DB에 기록합니다."""
#     query = text(
#         """
#         INSERT INTO interaction_logs (user_id, session_id, step_name, log_type, content)
#         VALUES (:user_id, :session_id, :step_name, :log_type, :content)
#         """
#     )
#     # DB에 저장하기 전에 content를 JSON 문자열로 변환합니다.
#     log_content = json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)
    
#     await db.execute(query, {
#         "user_id": user_id,
#         "session_id": session_id,
#         "step_name": step_name,
#         "log_type": log_type,
#         "content": log_content
#     })
#     # INSERT, UPDATE, DELETE 후에는 commit()을 호출해야 DB에 최종 반영됩니다.
#     await db.commit()


# services/db_service.py

import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Set, Optional

async def log_interaction(
    db: AsyncSession,
    session_id: str,
    step_name: str,
    log_type: str,
    content: dict | str,
    user_id: Optional[str] = None
):
    """(비동기) 사용자 상호작용 로그를 DB에 기록합니다."""
    query = text(
        """
        INSERT INTO interaction_logs (user_id, session_id, step_name, log_type, content)
        VALUES (:user_id, :session_id, :step_name, :log_type, :content)
        """
    )
    log_content = json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)
    
    await db.execute(query, {
        "user_id": user_id,
        "session_id": session_id,
        "step_name": step_name,
        "log_type": log_type,
        "content": log_content
    })
    await db.commit()

async def get_restaurants_from_db(db: AsyncSession, location_gu: str, amenities: List[str]) -> List[Dict]:
    """
    위치(구)와 편의시설 조건으로 DB에서 1차 후보 음식점 목록을 조회합니다.
    """
    print("\n[DEBUG] 1. DB에서 음식점 후보군 조회 시작")
    print(f"    - 입력 위치: {location_gu}, 편의시설: {amenities}")

    where_clauses = ["region_name LIKE :location_gu"]
    params = {"location_gu": f"%{location_gu}%"}

    amenity_map = {"주차": "has_parking", "와이파이": "has_wifi", "놀이방": "has_kids_zone"}
    for i, amenity in enumerate(amenities):
        if amenity in amenity_map:
            param_name = f"amenity_{i}"
            where_clauses.append(f"{amenity_map[amenity]} = :{param_name}")
            params[param_name] = 'Y'

    query_str = f"SELECT name, branch_name FROM restaurant WHERE {' AND '.join(where_clauses)} LIMIT 100"
    query = text(query_str)
    
    print(f"    - 생성된 SQL: {query_str}")
    print(f"    - SQL 파라미터: {params}")

    result = await db.execute(query, params)
    restaurants = [dict(row) for row in result.mappings()]
    
    print(f"    - 조회 결과: {len(restaurants)}개의 음식점 발견")
    return restaurants

async def get_normalized_menus_for_restaurants(db: AsyncSession, restaurants: List[Dict]) -> Dict[str, Set[str]]:
    """선별된 음식점 목록에 대해 정규화된 메뉴를 조회합니다."""
    print("\n[DEBUG] 2. 정규화된 메뉴 조회 시작")
    if not restaurants:
        print("    - 입력된 음식점이 없어 메뉴 조회를 건너뜁니다.")
        return {}

    restaurant_names = list(set(r['name'] for r in restaurants))
    params = {"names": restaurant_names}
    
    query = text("""
        SELECT restaurant_name, branch_name, normalized_name 
        FROM restaurant_menu 
        WHERE restaurant_name IN :names AND normalized_name IS NOT NULL
    """)
    
    print(f"    - 생성된 SQL: SELECT ... WHERE restaurant_name IN :names ...")
    print(f"    - SQL 파라미터 (names): {restaurant_names[:5]}... 등 {len(restaurant_names)}개")

    result = await db.execute(query, params)
    
    menus_by_restaurant = {}
    for row in result.mappings():
        full_name = f"{row['restaurant_name']} {row.get('branch_name', '') or ''}".strip()
        if full_name not in menus_by_restaurant:
            menus_by_restaurant[full_name] = set()
        menus_by_restaurant[full_name].add(row['normalized_name'])
    
    print(f"    - 조회 결과: {len(menus_by_restaurant)}개 음식점의 메뉴 정보 발견")
    return menus_by_restaurant