# core/config.py
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from typing import List, Dict, Set, Optional


load_dotenv()

class Settings:
    PROJECT_NAME: str = "AI Food Recommendation API"
    API_V1_STR: str = "/api/v1"

    # Security
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")

    # --- Database (변수 이름을 다른 파일과 통일) ---
    MYSQL_USER: str = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD_RAW = os.getenv("MYSQL_PASSWORD")
    # 비밀번호가 없는 경우를 대비한 안전장치 추가
    MYSQL_PASSWORD = quote_plus(MYSQL_PASSWORD_RAW) if MYSQL_PASSWORD_RAW else ""
    MYSQL_HOST: str = "155.248.175.96"
    MYSQL_PORT: str = os.getenv("MYSQL_PORT")
    MYSQL_DB_NAME: str = os.getenv("MYSQL_DB_NAME")
    DATABASE_URL = (
        f"mysql+aiomysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB_NAME}"
    )

    # OpenAI API Key
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    # --- ChromaDB Server ---
    CHROMA_HOST: str = os.getenv("CHROMA_HOST")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", 8000))


     # Data Paths for CSV files
    FOOD_CLUSTERS_PATH: str = os.getenv("FOOD_CLUSTERS_PATH")
    REPRESENTATIVE_FOODS_PATH: str = os.getenv("REPRESENTATIVE_FOODS_PATH")
    
    # Paths for local ChromaDB (data_loader에서 사용)
    HEALTH_JUDGMENT_DB_PATH: str = os.getenv("HEALTH_JUDGMENT_DB_PATH")
    REVIEW_DB_PATH: str = os.getenv("REVIEW_DB_PATH")
    
    # nlu_service에서 사용할 질병 키워드 맵
    DISEASE_KEYWORD_MAP: Dict[str, List[str]] ={
    "위식도 역류질환": ["위식도 역류질환", "GERD", "식도염", "식도역류증"],
    "위염 및 소화성 궤양": ["위염", "소화성 궤양", "위궤양"],
    "염증성 장질환": ["염증성 장질환", "IBD", "만성궤양성 대장염", "크론병"],
    "과민성 대장 증후군": ["과민성 대장 증후군", "IBS"],
    "변비 및 게실 질환": ["변비", "게실 질환", "게실염"],
    "간질환": ["간질환", "간염", "간경변", "B형 간염", "원발성 간암", "간암"],
    "담낭질환": ["담낭질환", "담석증", "담낭염"],
    "췌장염": ["췌장염", "Pancreatitis"],
    "고혈압 및 심혈관 질환": ["고혈압", "혈압", "심혈관", "심장", "동맥경화", "심부전", "심근경색증", "협심증", "관상동맥질환", "심장판막질환"],
    "고지혈증": ["고지혈증", "Hyperlipidemia", "콜레스테롤", "피가 탁함"],
    "뇌졸중": ["뇌졸중", "Stroke", "뇌경색", "뇌출혈", "중풍"],
    "당뇨병": ["당뇨병", "당뇨", "Diabetes", "혈당"],
    "통풍": ["통풍", "Gout"],
    "갑상선 질환": ["갑상선", "갑상선 기능 항진증", "갑상선 기능 저하증", "갑상선 결절"],
    "만성 신장질환": ["신장질환", "CKD", "신장", "콩팥", "신부전", "신증후군", "사구체콩팥염", "신장병"],
    "요로계 질환": ["요로계", "신결석", "요로결석", "요로감염"],
    "셀리악병": ["셀리악병", "Celiac Disease", "실리악 스푸루", "글루텐"],
    "유당 불내증": ["유당 불내증", "Lactose Intolerance"],
    "삼킴곤란": ["삼킴곤란", "연하곤란", "Dysphagia"],
    "빈혈": ["빈혈", "Anemia", "겸상 적혈구 빈혈증"],
    "암": ["암", "Cancer", "폐암", "백혈병", "유방암", "대장암", "림프종", "악성 종양"],
    "골격계 질환": ["골격계", "뼈", "골다공증", "골연화증", "구루병"],
    "알레르기": ["알레르기", "두드러기", "천식"],
    "급성 감염성 질환": ["감염", "감기", "폐렴", "코로나", "장염", "기관지염", "수두", "설사"],
}


settings = Settings()

# 디버깅을 위해 DB 연결 주소를 시작 시 출력
print("="*50)
print(f"✅ 사용되는 데이터베이스 주소: {settings.DATABASE_URL}")
print("="*50)
