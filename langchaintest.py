import pandas as pd
import os
import re
import time
import json
import pymysql
import shutil
import requests
import chromadb
import operator
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from typing import TypedDict, Optional, List

from chromadb import HttpClient
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import Document

from langgraph.graph import StateGraph, END

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ---------- MySQL 연결 설정 ----------
DB_CONFIG = {
    'host': '155.248.175.96',
    'port': 3306,
    'user': 'toy4_user',
    'password': os.getenv('MYSQL_PASSWORD'), 
    'database': os.getenv('MYSQL_DATABASE', 'nutrition_db'),
    'charset': 'utf8mb4'
}

# password 안전하게 인코딩
raw_password = DB_CONFIG['password']
if isinstance(raw_password, bytes):
    raw_password = raw_password.decode()

encoded_password = quote_plus(str(raw_password))


engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"
)

# ---------- Vector DB API 설정 ----------
VECTOR_DB_API_URL = "http://155.248.175.96:8000"
TARGET_COLLECTION_NAME = "disease_data"

# ChromaDB 클라이언트 초기화

chroma_client = HttpClient(
    host="155.248.175.96",
    port=8000
)
print("✅ ChromaDB HttpClient 성공적으로 초기화됨.")

# 2) 임베딩 함수 정의
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 3) LangChain Chroma 인스턴스 생성 (HTTP 모드)
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    client=chroma_client,
    collection_name=TARGET_COLLECTION_NAME,
    embedding_function=embedding
)
# 4) Retriever 초기화
retriever_nutrition = vectordb.as_retriever(search_kwargs={"k": 5})
print("✅ Retriever 준비 완료:", retriever_nutrition)


# try:
#     chroma_client = chromadb.HttpClient(host="155.248.175.96", port=8000)
#     print("ChromaDB HttpClient 성공적으로 초기화됨.")
# except Exception as e:
#     print(f"ChromaDB HttpClient 초기화 오류: {e}")
    
# --------------------연결 테스트
import requests
try:
    r = requests.get("http://155.248.175.96:8000/api/v2/heartbeat")
    if r.status_code == 200:
        print("✅ 서버 연결 정상:", r.json())
    else:
        print("⚠️ 연결 실패:", r.status_code)
except Exception as e:
    print("❌ 서버 연결 오류:", e)


# ---------- API 모델 ----------
class AskRequest(BaseModel):
    question: str

class DebugRequest(BaseModel):
    query: str
    doc_type: str  # 'vectordb' or 'nutrition'

# 피해야 할 음식 모델
class AvoidFood(BaseModel):
    name: str
    examples: List[str]

# 추천 음식 모델  
class RecommendedFood(BaseModel):
    name: str
    # calories_kcal: float
    # sodium_mg: int
    reason: str

# 응답 모델 수정
class DietRecommendationResponse(BaseModel):
    diseases: List[str]
    avoid_foods: List[AvoidFood]
    nutrition_standards: str
    recommended_foods: List[RecommendedFood]
    greeting: str
    status: str
    question: str
    debug_error: Optional[str] = None
    
# ---------- LangGraph State 정의 ----------
class DietRecommendationState(TypedDict):
    question: str
    diseases: List[str]
    avoid_foods: str
    nutrition_standards: str
    recommended_foods: str
    final_response: str
    greeting: str
    parsed_avoid_foods: List[AvoidFood]  # 추가
    parsed_recommended_foods: List[RecommendedFood]  # 추가
    current_step: str
    completed_steps: List[str]
    error: Optional[str]
    agent_logs: List[str]

def query_vector_db(query: str, k: int = 10):
    """ChromaDB Python 클라이언트를 사용하여 쿼리를 보내고 결과를 받아옴"""
    if chroma_client is None:
        print("ChromaDB 클라이언트가 초기화되지 않았습니다. 쿼리 불가.")
        return []

    try:
        try:
            target_collection = chroma_client.get_collection(name=TARGET_COLLECTION_NAME)
            print(f"사용할 컬렉션: {TARGET_COLLECTION_NAME}")
        except Exception as e:
            print(f"컬렉션 '{TARGET_COLLECTION_NAME}'을(를) 찾을 수 없습니다: {e}")
            return []
        
        results = target_collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        print(f"ChromaDB 쿼리 성공. 결과 수: {len(results.get('documents', []))}")
        
        return_docs = []
        if results and 'documents' in results and results['documents']:
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] else {}
                distance = results['distances'][0][i] if 'distances' in results and results['distances'] and results['distances'][0] else None
                
                return_docs.append({
                    "document": doc_content,
                    "metadata": metadata,
                    "distance": distance
                })
        return return_docs
            
    except Exception as e:
        print(f"Vector DB 쿼리 처리 오류 (ChromaDB 클라이언트): {e}")
        return []
    
# ---------- Rate Limit을 고려한 배치 임베딩 함수 ----------
def create_embeddings_with_retry(documents, batch_size=50, delay=5):
    """Rate Limit을 피하기 위해 배치로 임베딩 생성"""
    embedding = OpenAIEmbeddings()
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"배치 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} 처리 중... ({len(batch)}개 문서)")
        
        try:
            # 배치 임베딩 생성
            batch_embeddings = embedding.embed_documents([doc.page_content for doc in batch])
            all_embeddings.extend(batch_embeddings)
            
            # Rate Limit 방지를 위한 지연
            if i + batch_size < len(documents):
                print(f"{delay}초 대기 중...")
                time.sleep(delay)
                
        except Exception as e:
            print(f"배치 {i//batch_size + 1} 임베딩 실패: {e}")
            # Rate Limit 오류인 경우 더 긴 지연
            if "rate_limit" in str(e).lower():
                print("Rate Limit 감지. 30초 대기 후 재시도...")
                time.sleep(30)
                try:
                    batch_embeddings = embedding.embed_documents([doc.page_content for doc in batch])
                    all_embeddings.extend(batch_embeddings)
                except Exception as retry_error:
                    print(f"재시도 실패: {retry_error}")
                    # 실패한 배치는 빈 임베딩으로 처리
                    all_embeddings.extend([[0] * 1536] * len(batch))
            else:
                # 다른 오류는 빈 임베딩으로 처리
                all_embeddings.extend([[0] * 1536] * len(batch))
    
    return all_embeddings

# ---------- MySQL에서 식품 데이터 조회 및 텍스트 변환 ----------
def fetch_nutrition_data_from_mysql(limit=1000):
    """MySQL에서 식품 영양 정보를 조회하여 텍스트 형태로 변환"""
    try:
        # 새로운 컬럼 매핑
        column_mapping = {
            'name': '식품명',
            'energy_kcal': '에너지(kcal)',
            'moisture_g': '수분(g)',
            'protein_g': '단백질(g)',
            'fat_g': '지방(g)',
            'ash_g': '회분(g)',
            'carbohydrate_g': '탄수화물(g)',
            'sugar_g': '당류(g)',
            'dietary_fiber_g': '식이섬유(g)',
            'calcium_mg': '칼슘(mg)',
            'iron_mg': '철(mg)',
            'phosphorus_mg': '인(mg)',
            'potassium_mg': '칼륨(mg)',
            'sodium_mg': '나트륨(mg)',
            'vitamin_a_rae_ug': '비타민A(μg)',
            'retinol_ug': '레티놀(μg)',
            'beta_carotene_ug': '베타카토틴(μg)',
            'thiamine_mg': '티아민(mg)',
            'riboflavin_mg': '리보플라빈(mg)',
            'niacin_mg': '니아신(mg)',
            'vitamin_d_ug': '비타민D(μg)',
            'cholesterol_mg': '콜레스테롤(mg)',
            'saturated_fatty_acids_g': '포화지방산(g)',
            'trans_fat_g': '트랜스지방산(g)',
            'food_weight_g': '식품중량(g)'
        }
        
        # MySQL에서 데이터 조회
        query = f"""
        SELECT * FROM food_nutritional_ingredients 
        WHERE name IS NOT NULL 
        AND energy_kcal > 0
        ORDER BY RAND()
        LIMIT {limit}
        """
        
        df = pd.read_sql(query, engine)
        print(f"MySQL에서 조회된 식품 데이터: {len(df)}개")
        
    except Exception as e:
        print(f"MySQL 데이터 조회 오류: {e}")
        return []
    
# ---------- 동적 영양 조건 추출 및 쿼리 생성 함수들 ----------
def extract_nutrition_criteria_from_standards(nutrition_standards: str, disease: str) -> dict:
    """영양 기준 정보에서 MySQL 쿼리에 사용할 수치 조건들을 추출"""
    extract_criteria_prompt = PromptTemplate(
        input_variables=["nutrition_standards", "disease"],
        template="""아래 {disease} 환자의 영양 기준 정보에서 구체적인 수치(숫자와 단위)가 포함된 문장만 분석하여 JSON으로 추출하세요.

영양 기준 정보:
{nutrition_standards}

다음 영양소들에 대한 구체적인 수치 조건만 찾아서 JSON 형태로 출력하세요:
- sodium_mg: 나트륨 (mg 단위)
- sugar_g: 당류 (g 단위)  
- carbohydrate_g: 탄수화물 (g 단위)
- fat_g: 지방 (g 단위)
- cholesterol_mg: 콜레스테롤 (mg 단위)
- protein_g: 단백질 (g 단위)
- energy_kcal: 에너지 (kcal 단위)
- dietary_fiber_g: 식이섬유 (g 단위)

추출 규칙:
1. 반드시 수치(숫자)와 단위가 명시된 내용만 추출
2. "이내", "이하", "미만" 등은 최대값(max_)으로 처리
3. "이상", "최소" 등은 최소값(min_)으로 처리
4. 수치가 없거나 모호한 표현("적절히", "제한", "권장")은 무시
5. 빈 결과라면 {{}}로 출력

가능한 필드명:
- max_sodium_mg: 나트륨 최대 섭취량
- max_sugar_g: 당류 최대 섭취량  
- max_carbohydrate_g: 탄수화물 최대 섭취량
- max_cholesterol_mg: 콜레스테롤 최대 섭취량
- min_protein_g: 단백질 최소 섭취량
- max_fat_pct: 지방 최대 비율(%)
- min_fat_pct: 지방 최소 비율(%)

나쁜 예시:
- "나트륨 섭취를 제한" → 수치가 없으므로 포함하지 않음
- "적절한 단백질 섭취" → 수치가 없으므로 포함하지 않음

JSON 형태로만 출력:"""
    )
    
    try:
        from langchain.chains import LLMChain
        extract_chain = LLMChain(llm=llm, prompt=extract_criteria_prompt)
        result = extract_chain.run({
            "nutrition_standards": nutrition_standards,
            "disease": disease
        }).strip()
        
        print(f"\n추출된 영양 조건 (raw): {result}")
        
        # 1. JSON 블록 추출 (```json이나 ``` 태그 제거)
        cleaned_result = re.sub(r'```json\s*', '', result)
        cleaned_result = re.sub(r'```\s*', '', cleaned_result)
        cleaned_result = cleaned_result.strip()
        
        # 2. JSON 객체 부분만 추출
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_result, re.DOTALL)
        if json_match:
            json_str = json_match.group().strip()
            print(f"추출된 JSON 문자열: {json_str}")
            
            try:
                criteria = json.loads(json_str)
                print(f"파싱된 영양 조건: {criteria}")
                
                # 숫자가 아닌 값들 제거
                clean_criteria = {}
                for key, value in criteria.items():
                    if isinstance(value, (int, float)) and value > 0:
                        clean_criteria[key] = value
                    elif isinstance(value, str) and value.replace('.', '').isdigit():
                        clean_criteria[key] = float(value)
                
                print(f"정리된 영양 조건: {clean_criteria}")
                return clean_criteria
                
            except json.JSONDecodeError as je:
                print(f"JSON 파싱 오류: {je}")
                return {}
        else:
            print("JSON 형태를 찾을 수 없습니다.")
            return {}
            
    except Exception as e:
        print(f"영양 조건 추출 오류: {e}")
        return {}

def build_dynamic_mysql_query(criteria: dict) -> tuple:
    """추출된 영양 조건을 바탕으로 동적 MySQL 쿼리 생성"""
    base_query = """
    SELECT name, energy_kcal, protein_g, fat_g, carbohydrate_g, sugar_g, 
           sodium_mg, cholesterol_mg, calcium_mg, iron_mg, potassium_mg,
           dietary_fiber_g, vitamin_a_rae_ug
    FROM food_nutritional_ingredients 
    WHERE name IS NOT NULL AND energy_kcal > 0
    """
    
    conditions = []
    params = {}
    
    # 동적으로 조건 추가
    for key, value in criteria.items():
        if key.startswith("max_"):
            column_name = key[4:]  # "max_" 제거
            if column_name in ['sodium_mg', 'sugar_g', 'carbohydrate_g', 'fat_g', 
                             'cholesterol_mg', 'energy_kcal', 'dietary_fiber_g']:
                conditions.append(f"{column_name} <= :{key}")
                params[key] = value
                
        elif key.startswith("min_"):
            column_name = key[4:]  # "min_" 제거
            if column_name in ['protein_g', 'dietary_fiber_g', 'calcium_mg', 
                             'iron_mg', 'potassium_mg', 'vitamin_a_rae_ug']:
                conditions.append(f"{column_name} >= :{key}")
                params[key] = value
    
    # 조건이 있으면 추가
    if conditions:
        query = base_query + " AND " + " AND ".join(conditions) + " ORDER BY RAND() LIMIT 200"
    else:
        # 조건이 없으면 기본적으로 건강한 음식 위주로
        query = base_query + " AND sodium_mg < 500 ORDER BY RAND() LIMIT 200"
    
    print(f"생성된 쿼리: {query}")
    print(f"쿼리 파라미터: {params}")
    
    return query, params

def search_foods_by_dynamic_condition(criteria: dict):
    """동적으로 생성된 조건으로 음식 검색"""
    try:
        query, params = build_dynamic_mysql_query(criteria)
        df = pd.read_sql(text(query), engine, params=params)
        print(f"동적 조건으로 검색된 음식: {len(df)}개")
        return df
        
    except Exception as e:
        print(f"동적 조건 음식 검색 오류: {e}")
        return pd.DataFrame()

# ---------- 특정 조건으로 음식 검색 함수 ----------
def search_foods_by_condition(condition_type: str, **kwargs):
    """
    특정 조건에 맞는 음식을 MySQL에서 검색
    *** 하드코딩 부분 제거, 기본 조건만 유지 ***
    """
    try:
        base_query = """
        SELECT name, energy_kcal, protein_g, fat_g, carbohydrate_g, sugar_g, 
               sodium_mg, cholesterol_mg, calcium_mg, iron_mg, potassium_mg,
               dietary_fiber_g, vitamin_a_rae_ug
        FROM food_nutritional_ingredients 
        WHERE name IS NOT NULL AND energy_kcal > 0
        """
        
        # 기본 건강한 음식 조건만 유지 (fallback용)
        if condition_type == "healthy_default":
            query = base_query + " AND sodium_mg < 500 AND energy_kcal < 300 ORDER BY RAND() LIMIT 200"
        else:
            query = base_query + " ORDER BY RAND() LIMIT 200"
        
        df = pd.read_sql(query, engine)
        return df
        
    except Exception as e:
        print(f"조건별 음식 검색 오류: {e}")
        return pd.DataFrame()
    
# ---------- 영양 정보 데이터 준비 ----------
print("영양 정보 데이터 로딩 시작...")
nutrition_docs = fetch_nutrition_data_from_mysql(limit=500)  # 500개로 제한

if nutrition_docs:
    nutrition_text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separator="\n"
    )
    nutrition_docs = nutrition_text_splitter.split_documents(nutrition_docs)
    print(f"분할된 영양정보 청크: {len(nutrition_docs)}개")

# ---------- 벡터 DB (기존 벡터DB가 있으면 재사용, 없으면 생성) ----------
# embedding = OpenAIEmbeddings()
# vectordb_nutrition = None
# retriever_nutrition = vectordb_nutrition.as_retriever(search_kwargs={"k": 15})

# retriever_nutrition = None


# vector_db_path = "./vector_db_nutrition"




# # 기존 벡터 DB 확인
# if os.path.exists(vector_db_path) and os.listdir(vector_db_path):

# # ----------------- 경로 테스트--------------------------------------
#     vector_db_path = "./vector_db_nutrition"
#     print("현재 작업 디렉토리:", os.getcwd())
#     print("vector_db_path 경로 절대값:", os.path.abspath(vector_db_path))

#     print("경로 존재 여부:", os.path.exists(vector_db_path))  # True
#     print("디렉토리 내용물:", os.listdir(vector_db_path))  # 보기


# try:
        # print("기존 벡터 DB 로딩 중...")
        # vectordb_nutrition = Chroma(persist_directory=vector_db_path, embedding_function=embedding)
        # print("dddddddddddddddddddddddd")
        # print("기존 벡터 DB 로딩 완료")
# except Exception as e:
#     print(f"기존 벡터 DB 로딩 실패: {e}")
#     print("새로운 벡터 DB 생성을 진행합니다.")
        # if os.path.exists(vector_db_path):
        #     shutil.rmtree(vector_db_path)

# # 벡터 DB가 없거나 로딩 실패한 경우 새로 생성
# if vectordb_nutrition is None and nutrition_docs:
#     print("새로운 벡터 DB 생성 중...")
#     try:
#         # 배치로 나누어 처리
#         batch_size = 20  # 더 작은 배치 사이즈
#         total_batches = (len(nutrition_docs) - 1) // batch_size + 1
        
#         print(f"총 {len(nutrition_docs)}개 문서를 {total_batches}개 배치로 나누어 처리")
        
#         # 첫 번째 배치로 벡터 DB 초기화
#         first_batch = nutrition_docs[:batch_size]
#         vectordb_nutrition = Chroma.from_documents(
#             first_batch, 
#             embedding, 
#             persist_directory=vector_db_path
#         )
#         vectordb_nutrition.persist()
#         print(f"첫 번째 배치 ({len(first_batch)}개) 처리 완료")
        
#         # 나머지 배치들 순차 처리
#         for i in range(batch_size, len(nutrition_docs), batch_size):
#             batch = nutrition_docs[i:i+batch_size]
#             batch_num = i // batch_size + 1
            
#             print(f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 문서)")
            
#             try:
#                 vectordb_nutrition.add_documents(batch)
#                 vectordb_nutrition.persist()
                
#                 # Rate Limit 방지를 위한 지연
#                 if i + batch_size < len(nutrition_docs):
#                     print("Rate Limit 방지를 위해 10초 대기...")
#                     time.sleep(10)
                    
#             except Exception as batch_error:
#                 print(f"배치 {batch_num} 처리 실패: {batch_error}")
#                 if "rate_limit" in str(batch_error).lower():
#                     print("Rate Limit 감지. 30초 대기 후 재시도...")
#                     time.sleep(30)
#                     try:
#                         vectordb_nutrition.add_documents(batch)
#                         vectordb_nutrition.persist()
#                         print(f"배치 {batch_num} 재시도 성공")
#                     except Exception as retry_error:
#                         print(f"배치 {batch_num} 재시도 실패: {retry_error}")
#                         continue
        
#         retriever_nutrition = vectordb_nutrition.as_retriever(search_kwargs={"k": 15})
#         print("새로운 벡터 DB 생성 완료")
        
#     except Exception as e:
#         print(f"벡터 DB 생성 실패: {e}")
#         vectordb_nutrition = None
#         retriever_nutrition = None

# ---------- LLM 및 Chain 설정 ----------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=4096)

# ---------- Tool 함수들 ----------
def extract_disease_tool_func(question: str) -> str:
    extract_prompt = PromptTemplate(
        input_variables=["question"],
        template="""다음 사용자 질문에서 질병이나 건강 상태를 모두 리스트로 추출하세요.
    영어든 한국어든 상관없이 추출하세요. 형식은 쉼표로 구분된 한글 질병명만 나열합니다.  

예시:
질문: "당뇨가 있는데 저녁 추천해줘" → 답변: "당뇨"
질문: "고혈압 환자인데 뭘 먹을까?" → 답변: "고혈압"
질문: "나는 gout 가 있는데 저녁 메뉴 추천 해주셈" → 답변: "gout"
질문: "나는 간경변이 있는데 저녁 메뉴 추천 해주셈" → 답변: "간경변"
질문: "고혈압하고 통풍이 있어요" → "고혈압, 통풍"
질문: "당뇨, 고지혈증 모두 있는데 식단 알려줘" → "당뇨, 고지혈증"

질문: {question}
답변:"""
    )
    from langchain.chains import LLMChain
    extract_chain = LLMChain(llm=llm, prompt=extract_prompt)
    result = extract_chain.run({"question": question}).strip()
    print(f"\n추출된 질병명: {result}")
    return result

def get_avoid_foods_tool_func(disease: str) -> str:
    """Vector DB API를 사용하여 피해야 할 음식 정보 검색"""
    search_query_prompt = PromptTemplate(
        input_variables=["disease"],
        template="""질병명 "{disease}"를 기반으로 피해야 할 음식 정보를 찾기 위한 검색 쿼리를 생성하세요.

    다음을 포함하세요:
    - 질병명 한국어 표현
    - 질병명 영어 표현 
    - 피해야 할 음식, 금지 식품, 섭취 제한, 식이 조절, 나트륨/지방 제한 등 키워드
    - 한국어와 영어 키워드를 모두 포함

    질병명: {disease}
    생성할 검색 쿼리:"""
    )
    
    from langchain.chains import LLMChain
    query_chain = LLMChain(llm=llm, prompt=search_query_prompt)
    search_query = query_chain.run({"disease": disease}).strip()
    
    try:
        # Vector DB API 호출
        documents = query_vector_db(search_query, k=10)
        
        if not documents:
            return f"Vector DB에서 {disease}의 영양 기준 정보를 찾을 수 없습니다."
        
        # 문서 내용을 컨텍스트로 구성
        context = "\n".join([str(doc) for doc in documents])
        
        # LLM으로 답변 생성
        answer_prompt = PromptTemplate(
            input_variables=["context", "disease"],
            template="""
        아래 문서 내용을 참고해 "{disease}" 환자가 *꼭 피해야 하는 음식*을 최대한 의미 있는 카테고리로 구분하여 번호로 정렬해 요약하세요.

        제공된 문서:
        {context}
        
        규칙:
        1. "식품군"별로 묶어서 1~2단어로 깔끔하게 표현 (예: '염분이 높은 음식', '가공육', '기름진 음식' 등)
        2. 문서에서 반복 등장하는 음식·식품군은 1줄로 요약
        3. 각 문장 앞에는 번호를 붙이지 말고, 가능한 한 중복을 최소화하여 3~8개 항목으로 요약
        4. 각 항목 뒤에 괄호로 대표 예시를 함께 표기 (예: '젓갈, 소금, 된장' 등)
        5. 불필요한 부연설명, 의사상담 멘트 등은 쓰지 않는다

        피해야 할 음식 목록:
        """
        )
        
        answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
        result = answer_chain.run({"context": context, "disease": disease})
        
        return result
            
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"
    
def get_nutrition_standards_tool_func(disease: str) -> str:
    """Vector DB API를 사용하여 영양 기준 정보 검색"""
    search_query_prompt = PromptTemplate(
        input_variables=["disease"],
        template="""질병명 "{disease}"를 기반으로 영양 기준 정보를 찾기 위한 검색 쿼리를 생성하세요.

다음을 포함하세요:
- 질병명 한국어 표현
- 질병명 영어 표현
- 권장 섭취량, 식이요법, 영양 기준, 영양소 권장량, 식이지침 등 키워드
- 한국어와 영어 키워드를 모두 포함

질병명: {disease}
생성할 검색 쿼리:"""
    )
    
    from langchain.chains import LLMChain
    query_chain = LLMChain(llm=llm, prompt=search_query_prompt)
    search_query = query_chain.run({"disease": disease}).strip()
    
    try:
        # Vector DB API 호출
        documents = query_vector_db(search_query, k=10)
        
        if not documents:
            return f"Vector DB에서 {disease}의 영양 기준 정보를 찾을 수 없습니다."
        
        # 문서 내용을 컨텍스트로 구성
        context = "\n".join([str(doc) for doc in documents])
        
        # LLM으로 답변 생성
        answer_prompt = PromptTemplate(
            template="""**중요: 아래 제공된 문서 내용만을 사용하여 답변하세요. 문서가 영어인 경우 한국어로 번역하여 답변하세요.**

제공된 문서:
{context}

질문: {disease} 환자의 영양 섭취 기준은 무엇인가요?

답변 규칙:
1. 위 문서에 명시된 내용만 사용
2. 영어 문서의 경우 한국어로 번역하여 답변
3. 최소한의 단어로 깔끔하게 표현
4. 각 문장 앞에는 번호를 붙이지 말고,문서에서 반복 등장하는 영양 성분은 1줄로 요약
5. 문서에 직접적으로 명시되어 있지 않더라도 관련 정보가 있으면 간접적으로 표현 가능. 단, 문서에 기반한 추론임을 명시하세요.
6. 불필요한 부연설명, 의사상담 멘트 금지

답변:""",
            input_variables=["context", "disease"]
        )
        
        answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
        result = answer_chain.run({"context": context, "disease": disease})
        
        return result
            
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"

def recommend_foods_tool_func(input_str: str) -> str:
    """
    동적 조건 생성 사용
    """
    if not retriever_nutrition:
        return "영양 데이터가 없어 음식을 추천할 수 없습니다."
    
    # 입력 파싱
    parts = input_str.split("|")
    disease = parts[0] if len(parts) > 0 else ""
    avoid_foods = parts[1] if len(parts) > 1 else ""
    nutrition_standards = "|".join(parts[2:]) if len(parts) > 2 else ""
    
    if not disease:
        return "질병명이 제공되지 않아 추천이 불가능합니다."
    
    print(f"\n동적 음식 추천 - 질병: {disease}")
    print(f"피해야할음식: {avoid_foods}")
    print(f"영양기준: {nutrition_standards[:200]}...")
    
    try:
        # 1. 영양 기준에서 수치 조건 추출 (새로운 동적 방식)
        criteria = extract_nutrition_criteria_from_standards(nutrition_standards, disease)
        
        if not criteria:
            print("추출된 영양 조건이 없어서 기본 건강식 조건을 사용합니다.")
            # 기본 조건으로 대체
            condition_foods = search_foods_by_condition('healthy_default')
        else:
            # 2. 동적 조건으로 음식 검색
            condition_foods = search_foods_by_dynamic_condition(criteria)
        
        if condition_foods.empty:
            # Vector DB 검색으로 대체
            health_query = f"{disease} 건강식 추천 음식"
            docs = retriever_nutrition.get_relevant_documents(health_query)
            context = "\n".join([doc.page_content for doc in docs])
        else:
            # MySQL 결과를 텍스트로 변환
            context = f"질병: {disease}\n피해야 할 음식: {avoid_foods}\n영양 기준: {nutrition_standards}\n"
            context += f"적용된 영양 조건: {criteria}\n\n"
            context += "추천 가능한 음식들:\n"
            
            for idx, row in condition_foods.head(100).iterrows():
                food_info = f"{idx+1}. {row['name']} | "
                food_info += f"칼로리: {row['energy_kcal']:.1f}kcal | "
                food_info += f"단백질: {row['protein_g']:.1f}g | "
                food_info += f"지방: {row['fat_g']:.1f}g | "
                food_info += f"탄수화물: {row['carbohydrate_g']:.1f}g | "
                food_info += f"나트륨: {int(row['sodium_mg'])}mg"
                
                # 추가 영양소 정보
                if pd.notna(row['sugar_g']):
                    food_info += f" | 당류: {row['sugar_g']:.1f}g"
                if pd.notna(row['cholesterol_mg']):
                    food_info += f" | 콜레스테롤: {int(row['cholesterol_mg'])}mg"
                if pd.notna(row['dietary_fiber_g']):
                    food_info += f" | 식이섬유: {row['dietary_fiber_g']:.1f}g"
                    
                context += food_info + "\n"
        
        # 3. LLM으로 최종 추천 생성
        recommendation_prompt = PromptTemplate(
            template="""다음 정보를 바탕으로 {disease} 환자에게 적합한 음식을 최대한 많이 추천하세요.

{context}

**중요한 추천 규칙:**
1. 위에 제공된 음식 목록에서 영양 조건에 맞는 음식들을 최대한 많이 선택하여 추천하세요 (50개 이상 목표)
2. 피해야 할 음식: "{avoid_foods}"에 포함된 것은 절대 추천하지 마세요
3. 적용된 영양 조건을 준수하는 음식만 선택하세요
4. 각 음식마다 번호를 매기고 다음 형식으로 작성하세요:
   "번호. 음식명 - 칼로리: XX kcal, 주요영양소 정보 (추천 이유: {disease}에 좋은 이유)"
5. {disease} 환자에게 왜 좋은지 각 음식마다 간단히 설명하세요
6. 제공된 데이터에 있는 음식만 추천하세요
7. 다양한 카테고리의 음식을 포함하세요 (주식, 반찬, 국물요리, 간식 등)
8. 영양 조건에 특히 잘 맞는 음식들을 우선적으로 추천하세요

**목표: 최소 30개 이상, 가능하면 50개 이상의 음식을 추천하세요**

{disease} 환자를 위한 맞춤 추천 음식 목록:""",
            input_variables=["disease", "context", "avoid_foods"]
        )
        
        from langchain.chains import LLMChain
        recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)
        result = recommendation_chain.run({
            "disease": disease,
            "context": context,
            "avoid_foods": avoid_foods
        })
        
        return result
        
    except Exception as e:
        print(f"동적 음식 추천 오류: {e}")
        return f"추천 중 오류가 발생했습니다: {str(e)}"
    
# ---------- Tool 목록 ----------
tools = [
    Tool(
        name="ExtractDisease",
        func=extract_disease_tool_func,
        description="사용자 질문에서 질병명이나 건강 상태를 추출합니다."
    ),
    Tool(
        name="GetAvoidFoods",
        func=get_avoid_foods_tool_func,
        description="질병별로 *반드시 피해야 할 음식 목록만* 조회합니다."
    ),
    Tool(
        name="GetNutritionStandards",
        func=get_nutrition_standards_tool_func,
        description="질병별로 *권장 영양 섭취 기준만* 조회합니다."
    ),
    Tool(
        name="RecommendFoods",
        func=recommend_foods_tool_func,
        description="MySQL 데이터베이스에서 질병 조건에 맞는 적합한 음식을 추천합니다."
    )
]

def create_agent_with_tools():
    """Tool들을 사용하는 Agent 생성"""
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )
    return agent

# Agent 인스턴스 생성
diet_agent = create_agent_with_tools()

# ---------- LangGraph Node 함수들 ----------
def extract_disease_node(state: DietRecommendationState) -> DietRecommendationState:
    """질병 추출 노드 - Tool 함수 직접 호출"""
    try:
        question = state["question"]
        print(f"질병 추출 단계: {question}")
        result = extract_disease_tool_func(question)
        diseases = [d.strip() for d in result.split(",") if d.strip()]
        
        return {
            **state,
            "diseases": diseases,
            "current_step": "extract_disease",
            "completed_steps": ["extract_disease"],
            "agent_logs": [f"질병 추출 완료: {diseases}"]
        }
    except Exception as e:
        return {
            **state,
            "error": f"질병 추출 오류: {str(e)}",
            "current_step": "error"
        }

def get_avoid_foods_node(state: DietRecommendationState) -> DietRecommendationState:
    """피해야 할 음식 조회 노드 - Tool 함수 직접 호출"""
    try:
        diseases = state.get("diseases", [])
        if not diseases:
            return {
                **state,
                "error": "추출된 질병이 없습니다.",
                "current_step": "error"
            }
        
        avoid_foods_results = []
        prev_logs = state.get("agent_logs", [])
        prev_completed = state.get("completed_steps", [])
        
        for disease in diseases:
            print(f"피해야 할 음식 조회: {disease}")
            result = get_avoid_foods_tool_func(disease)
            avoid_foods_results.append(result)
        
        avoid_foods = "\n\n".join(avoid_foods_results)
        
        new_logs = prev_logs + [f"피해야 할 음식 조회 완료: {', '.join(diseases)}"]
        new_completed = prev_completed + ["get_avoid_foods"]
        
        return {
            **state,
            "avoid_foods": avoid_foods,
            "current_step": "get_avoid_foods",
            "completed_steps": new_completed,
            "agent_logs": new_logs
        }
    except Exception as e:
        return {
            **state,
            "error": f"피해야 할 음식 조회 오류: {str(e)}",
            "current_step": "error"
        }

def get_nutrition_standards_node(state: DietRecommendationState) -> DietRecommendationState:
    """영양 기준 조회 노드 - Tool 함수 직접 호출 사용"""
    try:
        diseases = state.get("diseases", [])
        if not diseases:
            return {
                **state,
                "error": "추출된 질병이 없습니다.",
                "current_step": "error"
            }
        
        nutrition_results = []
        prev_logs = state.get("agent_logs", [])
        prev_completed = state.get("completed_steps", [])
        
        for disease in diseases:
            print(f"영양 기준 조회: {disease}")
            result = get_nutrition_standards_tool_func(disease)
            nutrition_results.append(result)
        
        nutrition_standards = "\n\n".join(nutrition_results)
        
        new_logs = prev_logs + [f"영양 기준 조회 완료: {', '.join(diseases)}"]
        new_completed = prev_completed + ["get_nutrition_standards"]
        
        return {
            **state,
            "nutrition_standards": nutrition_standards,
            "current_step": "get_nutrition_standards",
            "completed_steps": new_completed,
            "agent_logs": new_logs
        }
    except Exception as e:
        return {
            **state,
            "error": f"영양 기준 조회 오류: {str(e)}",
            "current_step": "error"
        }

def recommend_foods_node(state: DietRecommendationState) -> DietRecommendationState:
    """음식 추천 노드 - Tool 함수 직접 호출"""
    try:
        diseases = state.get("diseases", [])
        avoid_foods = state.get("avoid_foods", "")
        nutrition_standards = state.get("nutrition_standards", "")
        
        print(f"DEBUG - recommend_foods_node 시작:")
        print(f"  diseases: {diseases}")
        print(f"  avoid_foods: {avoid_foods[:100]}...")
        print(f"  nutrition_standards: {nutrition_standards[:100]}...")
        
        if not diseases:
            return {
                **state,
                "error": "추출된 질병이 없습니다.",
                "current_step": "error"
            }
            
        # 필수 입력 확인
        if not avoid_foods or not nutrition_standards:
            return {
                **state,
                "error": "피해야 할 음식 또는 영양 기준 정보가 없습니다.",
                "current_step": "error"
            }
        
        recommended_results = []
        prev_logs = state.get("agent_logs", [])
        prev_completed = state.get("completed_steps", [])
        
        for disease in diseases:
            print(f"음식 추천 시작: {disease}")
            input_str = f"{disease}|{avoid_foods}|{nutrition_standards}"
            result = recommend_foods_tool_func(input_str)
            print(f"추천 결과 길이: {len(result)} 글자")
            print(f"추천 결과 미리보기: {result[:200]}...")
            recommended_results.append(result)
        
        recommended_foods = "\n\n".join(recommended_results)
        print(f"DEBUG - 최종 recommended_foods 길이: {len(recommended_foods)}")

        new_logs = prev_logs + [f"음식 추천 완료: {', '.join(diseases)}"]
        new_completed = prev_completed + ["recommend_foods"]
        
        updated_state = {
            **state,
            "recommended_foods": recommended_foods,
            "current_step": "recommend_foods",
            "completed_steps": new_completed,
            "agent_logs": new_logs
        }
        
        # 상태 저장 확인
        print(f"DEBUG - 상태 저장 확인:")
        print(f"  recommended_foods in state: {len(updated_state.get('recommended_foods', ''))}")
        
        return updated_state
        
    except Exception as e:
        print(f"음식 추천 오류: {e}")
        return {
            **state,
            "error": f"음식 추천 오류: {str(e)}",
            "current_step": "error"
        }
        
def parse_avoid_foods(avoid_foods_text: str) -> List[AvoidFood]:
    """피해야 할 음식 텍스트를 구조화된 리스트로 파싱"""
    avoid_foods = []
    
    # 번호로 시작하는 라인들을 찾기
    lines = avoid_foods_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+\.', line):
            # "1. 고염분 음식 (소금, 가공식품)" 형태 파싱
            match = re.match(r'^\d+\.\s*([^(]+)\s*\(([^)]+)\)', line)
            if match:
                name = match.group(1).strip()
                examples_str = match.group(2).strip()
                examples = [ex.strip() for ex in examples_str.split(',')]
                
                avoid_foods.append(AvoidFood(
                    name=name,
                    examples=examples
                ))
    
    return avoid_foods

def parse_recommended_foods(recommended_foods_text: str) -> List[RecommendedFood]:
    """추천 음식 텍스트를 구조화된 리스트로 파싱"""
    recommended_foods = []
    
    # 번호로 시작하는 각 음식 항목을 찾기
    lines = recommended_foods_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+\.', line):
            try:
                # "1. 단무지무침 - 칼로리: 41.0kcal, 나트륨: 611mg (고혈압에 좋은 이유: ...)" 형태 파싱
                
                # 음식명 추출
                name_match = re.match(r'^\d+\.\s*([^-]+)', line)
                if not name_match:
                    continue
                name = name_match.group(1).strip()
                
                # 칼로리 추출
                calorie_match = re.search(r'칼로리:\s*([\d.]+)kcal', line)
                calories = float(calorie_match.group(1)) if calorie_match else 0.0
                
                # 나트륨 추출
                sodium_match = re.search(r'나트륨:\s*(\d+)mg', line)
                sodium = int(sodium_match.group(1)) if sodium_match else 0
                
                # 추천 이유 추출
                reason_match = re.search(r'\(고혈압에 좋은 이유:\s*([^)]+)\)', line)
                reason = reason_match.group(1).strip() if reason_match else ""
                
                recommended_foods.append(RecommendedFood(
                    name=name,
                    # calories_kcal=calories,
                    # sodium_mg=sodium,
                    reason=reason
                ))
                
            except Exception as e:
                print(f"음식 파싱 오류 - {line}: {e}")
                continue
    
    return recommended_foods

def generate_final_response_node(state: DietRecommendationState) -> DietRecommendationState:
    """최종 응답 생성 노드 - 구조화된 데이터 준비"""
    try:
        diseases = state.get("diseases", [])
        avoid_foods_text = state.get("avoid_foods", "").strip()
        nutrition_standards = state.get("nutrition_standards", "").strip()
        recommended_foods_text = state.get("recommended_foods", "").strip()

        # 디버깅 로그 추가
        print(f"DEBUG - generate_final_response_node:")
        print(f"  diseases: {diseases}")
        print(f"  avoid_foods length: {len(avoid_foods_text)}")
        print(f"  nutrition_standards length: {len(nutrition_standards)}")
        print(f"  recommended_foods length: {len(recommended_foods_text)}")

        # 인사말 생성
        greeting = f"안녕하세요! {', '.join(diseases)} 환자님을 위한 식단을 안내해 드립니다."

        # 텍스트를 구조화된 데이터로 파싱
        parsed_avoid_foods = parse_avoid_foods(avoid_foods_text) if avoid_foods_text else []
        parsed_recommended_foods = parse_recommended_foods(recommended_foods_text) if recommended_foods_text else []

        print(f"DEBUG - Parsed avoid_foods: {len(parsed_avoid_foods)}개")
        print(f"DEBUG - Parsed recommended_foods: {len(parsed_recommended_foods)}개")

        # 기존 final_response는 호환성을 위해 유지
        sections = [greeting]
        if avoid_foods_text:
            sections.append("===== 피해야 할 음식 =====")
            sections.append(avoid_foods_text)
        if nutrition_standards:
            sections.append("\n===== 영양 섭취 기준 =====")
            sections.append(nutrition_standards)
        if recommended_foods_text:
            sections.append("\n===== 추천 음식 =====")
            sections.append(recommended_foods_text)

        final_response = "\n".join(sections)

        # 상태 누적 (불변성 보장)
        prev_logs = state.get("agent_logs", [])
        prev_completed = state.get("completed_steps", [])
        new_logs = prev_logs + ["최종 응답 생성 완료"]
        new_completed = prev_completed + ["generate_final_response"]

        return {
            **state,
            "final_response": final_response,
            "greeting": greeting,
            "parsed_avoid_foods": parsed_avoid_foods,  # 파싱된 데이터 추가
            "parsed_recommended_foods": parsed_recommended_foods,  # 파싱된 데이터 추가
            "current_step": "completed",
            "completed_steps": new_completed,
            "agent_logs": new_logs
        }
    except Exception as e:
        print(f"ERROR in generate_final_response_node: {e}")
        return {
            **state,
            "error": f"최종 응답 생성 오류: {e}",
            "current_step": "error"
        }
        
def error_node(state: DietRecommendationState) -> DietRecommendationState:
    """에러 처리 노드"""
    error_message = state.get("error", "알 수 없는 오류가 발생했습니다.")
    return {
        **state,
        "final_response": f"죄송합니다. 처리 중 오류가 발생했습니다: {error_message}",
        "current_step": "error_handled"
    }

# ---------- 조건부 라우팅 함수 ----------
def should_continue(state: DietRecommendationState) -> str:
    """다음 단계 결정 (상태 전달 디버깅 포함)"""
    current_step = state.get("current_step", "")
    completed_steps = state.get("completed_steps", [])
    
    # 디버깅을 위한 로그
    print(f"DEBUG: current_step = {current_step}")
    print(f"DEBUG: completed_steps length = {len(completed_steps)}")
    print(f"DEBUG: completed_steps = {completed_steps}")
    
    # 상태 전달 확인 - 특히 recommended_foods
    if "recommended_foods" in state:
        recommended_foods_len = len(str(state["recommended_foods"]))
        print(f"DEBUG: should_continue에서 recommended_foods 길이 = {recommended_foods_len}")
        if recommended_foods_len > 0:
            preview = str(state["recommended_foods"])[:100]
            print(f"DEBUG: recommended_foods 미리보기 = {preview}...")
    else:
        print("DEBUG: recommended_foods 키가 state에 없습니다!")
    
    # 전체 state 키 확인
    print(f"DEBUG: should_continue에서 전체 state 키 = {list(state.keys())}")
    print("retriever_nutrition:", retriever_nutrition)
    print("type:", type(retriever_nutrition))
    print("retriever_nutrition vectorstore:", type(retriever_nutrition.vectorstore))
    print("retriever_nutrition vectorstore base:", retriever_nutrition.vectorstore._collection if hasattr(retriever_nutrition.vectorstore, '_collection') else 'N/A')
    
    # client = vectordb._client
    # if isinstance(client, HttpClient):
    #     print("✅ HTTP Chroma Vector DB 사용 중!")
    # else:
    #     print(" ❗️로컬 Chroma DB 사용 중!")
    
    if current_step == "error":
        return "error"
    elif current_step == "extract_disease":
        return "get_avoid_foods"
    elif current_step == "get_avoid_foods":
        return "get_nutrition_standards"
    elif current_step == "get_nutrition_standards":
        return "recommend_foods"
    elif current_step == "recommend_foods":
        return "generate_final_response"
    elif current_step == "completed":
        return END
    elif current_step == "error_handled":
        return END
    else:
        print(f"WARNING: Unexpected current_step: {current_step}")
        return "extract_disease"

# ---------- LangGraph 생성 ----------
def create_diet_recommendation_graph():
    """식단 추천 그래프 생성"""
    workflow = StateGraph(DietRecommendationState)
    
    # 노드 추가
    workflow.add_node("extract_disease", extract_disease_node)
    workflow.add_node("get_avoid_foods", get_avoid_foods_node)
    workflow.add_node("get_nutrition_standards", get_nutrition_standards_node)
    workflow.add_node("recommend_foods", recommend_foods_node)
    workflow.add_node("generate_final_response", generate_final_response_node)
    workflow.add_node("error", error_node)
    
    # 시작점 설정
    workflow.set_entry_point("extract_disease")
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "extract_disease",
        should_continue,
        {
            "get_avoid_foods": "get_avoid_foods",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "get_avoid_foods",
        should_continue,
        {
            "get_nutrition_standards": "get_nutrition_standards",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "get_nutrition_standards",
        should_continue,
        {
            "recommend_foods": "recommend_foods",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "recommend_foods",
        should_continue,
        {
            "generate_final_response": "generate_final_response",
            "error": "error"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_final_response",
        should_continue,
        {
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "error",
        should_continue,
        {
            END: END
        }
    )
    
    return workflow.compile()

# 전역 그래프 인스턴스 생성
diet_recommendation_graph = create_diet_recommendation_graph()
    
@app.get("/")
def root():
    return {
        "message": "MySQL 연동 RAG 기반 식단 추천 API 서버 실행 중",
        "vector_db_api": VECTOR_DB_API_URL,
        "nutrition_docs": len(nutrition_docs) if nutrition_docs else 0,
        "mysql_connection": "연결됨" if engine else "연결 안됨"
    }
    
            
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


@app.post("/ask", response_model=DietRecommendationResponse)
def ask(request: AskRequest):
    try:
        print(f"받은 질문: {request.question}")
        
        # LangGraph 사용
        initial_state = {
            "question": request.question,
            "diseases": [],
            "avoid_foods": "",
            "nutrition_standards": "",
            "recommended_foods": "",
            "final_response": "",
            "greeting": "",
            "parsed_avoid_foods": [],  # 추가
            "parsed_recommended_foods": [],  # 추가
            "current_step": "extract_disease",
            "completed_steps": [],
            "error": None,
            "agent_logs": []
        }
        
        final_state = diet_recommendation_graph.invoke(initial_state)
        
        return DietRecommendationResponse(
            diseases=final_state.get("diseases", []),
            avoid_foods=final_state.get("parsed_avoid_foods", []),
            nutrition_standards=final_state.get("nutrition_standards", ""),
            recommended_foods=final_state.get("parsed_recommended_foods", []),
            greeting=final_state.get("greeting", ""),
            status="success" if not final_state.get("error") else "error",
            question=request.question
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"에러 발생: {error_msg}")
        
        return DietRecommendationResponse(
            diseases=[],
            avoid_foods=[],
            nutrition_standards="",
            recommended_foods=[],
            greeting="",
            status="error",
            question=request.question,
            debug_error=error_msg
        )

# ---------- 호환성을 위한 기존 API도 유지 ----------
@app.post("/ask-simple")
def ask_simple(request: AskRequest):
    """기존 방식의 단순한 응답 (호환성 유지용)"""
    try:
        print(f"받은 질문: {request.question}")
        
        # LangGraph 사용
        initial_state = {
            "question": request.question,
            "diseases": [],
            "avoid_foods": "",
            "nutrition_standards": "",
            "recommended_foods": "",
            "final_response": "",
            "greeting": "",
            "parsed_avoid_foods": [],
            "parsed_recommended_foods": [],
            "current_step": "extract_disease",
            "completed_steps": [],
            "error": None,
            "agent_logs": []
        }
        
        final_state = diet_recommendation_graph.invoke(initial_state)
        
        return {
            "response": final_state.get("final_response", "응답 생성에 실패했습니다."),
            "status": "success" if not final_state.get("error") else "error",
            "question": request.question
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"에러 발생: {error_msg}")
        
        return {
            "error": "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "status": "error",
            "question": request.question,
            "debug_error": error_msg
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"에러 발생: {error_msg}")
        
        return DietRecommendationResponse(
            diseases=[],
            avoid_foods="",
            nutrition_standards="",
            recommended_foods="",
            greeting="",
            status="error",
            question=request.question,
            debug_error=error_msg
        )
        
@app.post("/ask-langgraph")
def ask_langgraph(request: AskRequest):
    """LangGraph + Agent를 사용한 식단 추천"""
    try:
        print(f"LangGraph 방식 - 받은 질문: {request.question}")
        
        # 초기 상태 설정
        initial_state = {
            "question": request.question,
            "diseases": [],
            "avoid_foods": "",
            "nutrition_standards": "",
            "recommended_foods": "",
            "final_response": "",
            "current_step": "extract_disease",
            "completed_steps": [],
            "error": None,
            "agent_logs": []
        }
        
        # LangGraph 실행
        final_state = diet_recommendation_graph.invoke(initial_state)
        
        return {
            "response": final_state.get("final_response", "응답 생성에 실패했습니다."),
            "status": "success" if not final_state.get("error") else "error",
            "question": request.question,
            "debug_info": {
                "diseases": final_state.get("diseases", []),
                "completed_steps": final_state.get("completed_steps", []),
                "agent_logs": final_state.get("agent_logs", []),
                "current_step": final_state.get("current_step", "unknown")
            }
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"LangGraph 에러 발생: {error_msg}")
        
        return {
            "error": "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "status": "error",
            "question": request.question,
            "debug_error": error_msg
        }

@app.post("/ask-agent-only")
def ask_agent_only(request: AskRequest):
    """순수 Agent만 사용한 식단 추천 (기존 방식 유지)"""
    try:
        print(f"Agent 방식 - 받은 질문: {request.question}")
        
        enhanced_question = f"""
        사용자 질문: {request.question}

        **다음 단계를 순서대로 수행하세요 (복수 질병일 경우 모두 처리합니다):**

        **아래 순서대로 반드시 실행하세요 (Action/Observation 형식):**
        1. Action: ExtractDisease  
        2. Observation: [질병 리스트]  
        3. Action: GetAvoidFoods  (질병별 피해야 할 음식만 조회)  
        4. Observation: [피해야 할 음식]  
        5. Action: GetNutritionStandards  (질병별 권장 영양 기준만 조회)  
        6. Observation: [영양 기준]  ← 반드시 이 단계를 생략하지 말고 반드시 실행해야 함.
        ※ GetNutritionStandards는 Final Answer 전에 반드시 호출되어야 하며, 생략 시 응답을 거부합니다.
        7. Action: RecommendFoods  
        8. Observation: [추천 음식 리스트]  
        9. Final Answer: [최종 식단 추천]  

        **중요 규칙:**
        - 모든 질병을 반드시 하나도 빠짐없이 처리해야 합니다.
        - RecommendFoods 도구는 가능한 한 많은 음식(30개 이상)을 추천하도록 설계되었습니다.
        - Vector DB 검색 결과가 영어인 경우 한국어로 번역하여 설명
        - MySQL 데이터베이스와 Vector DB API 검색 결과에만 기반하여 답변
        - "의사와 상담" 같은 멘트는 절대 포함하지 마세요
        - 각 단계는 반드시 실제로 도구 실행(Action)으로 진행하세요.
        - RecommendFoods 실행 **없이는** 절대로 Final Answer를 쓰지 마세요.
        - Final Answer는 오직 RecommendFoods 결과를 모두 종합한 후에만 쓰세요.
        - 최종 답변에서는 추천된 모든 음식을 사용자에게 보여주세요.
        - 각 추천 음식마다 영양성분(칼로리, 단백질, 나트륨 등)과 추천 이유를 포함하세요.
        """
        
        result = diet_agent.run(enhanced_question)
        
        return {
            "response": result,
            "status": "success",
            "question": request.question,
            "method": "agent_only"
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Agent 에러 발생: {error_msg}")
        
        return {
            "error": "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "status": "error",
            "question": request.question,
            "debug_error": error_msg
        }
