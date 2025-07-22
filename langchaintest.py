import pandas as pd
import os
import time
import pymysql
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.schema import Document
from urllib.parse import quote_plus

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ---------- MySQL 연결 설정 ----------
DB_CONFIG = {
    'host': '155.248.175.96',
    'port': 3306,
    'user': 'toy4_user',
    'password': os.getenv('MYSQL_PASSWORD'), 
    'database': os.getenv('MYSQL_DATABASE', 'nutrition_db'),  # 데이터베이스명
    'charset': 'utf8mb4'
}

encoded_password = quote_plus(DB_CONFIG['password'])

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"
)

# ---------- MySQL에서 식품 데이터 조회 및 텍스트 변환 ----------
def fetch_nutrition_data_from_mysql():
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
        query = """
        SELECT name, energy_kcal, moisture_g, protein_g, fat_g, ash_g, carbohydrate_g, sugar_g, dietary_fiber_g, calcium_mg, iron_mg, phosphorus_mg, 
               potassium_mg, sodium_mg, vitamin_a_rae_ug, retinol_ug, beta_carotene_ug, thiamine_mg, riboflavin_mg, niacin_mg, vitamin_d_ug, 
               cholesterol_mg, saturated_fatty_acids_g, trans_fat_g, food_weight_g
        FROM food_nutritional_ingredients 
        WHERE name IS NOT NULL 
        AND energy_kcal > 0
        ORDER BY RAND()
        LIMIT 1000
        """
        
        df = pd.read_sql(query, engine)
        print(f"MySQL에서 조회된 식품 데이터: {len(df)}개")
        
        # 텍스트 문서로 변환
        documents = []
        for _, row in df.iterrows():
            items = []
            for col, korean_name in column_mapping.items():
                if col in df.columns and pd.notna(row[col]) and row[col] != '':
                    val = row[col]
                    if isinstance(val, (int, float)) and val > 0:
                        if col in ['energy_kcal', 'protein_g', 'fat_g', 'carbohydrate_g', 'dietary_fiber_g']:
                            formatted_val = f"{val:.1f}"
                        elif col in ['sodium_mg', 'cholesterol_mg', 'calcium_mg', 'iron_mg', 'potassium_mg']:
                            formatted_val = f"{int(val)}"
                        elif 'vitamin' in col or 'thiamine' in col or 'riboflavin' in col or 'niacin' in col:
                            formatted_val = f"{val:.2f}" if val < 1 else f"{val:.1f}"
                        else:
                            formatted_val = str(val)
                        items.append(f"{korean_name}: {formatted_val}")
            
            if len(items) >= 3:  # 최소 3개 이상의 영양소 정보가 있는 경우만 포함
                content = " | ".join(items)
                doc = Document(page_content=content, metadata={"source": "mysql_nutrition"})
                documents.append(doc)
        
        print(f"변환된 문서: {len(documents)}개")
        return documents
        
    except Exception as e:
        print(f"MySQL 데이터 조회 오류: {e}")
        return []

# ---------- 특정 조건으로 음식 검색 함수 ----------
def search_foods_by_condition(condition_type: str, **kwargs):
    """특정 조건에 맞는 음식을 MySQL에서 검색"""
    try:
        base_query = """
        SELECT name, energy_kcal, protein_g, fat_g, carbohydrate_g, sugar_g, 
               sodium_mg, cholesterol_mg, calcium_mg, iron_mg, potassium_mg,
               dietary_fiber_g, vitamin_a_rae_ug
        FROM food_nutritional_ingredients 
        WHERE name IS NOT NULL AND energy_kcal > 0
        """
        
        conditions = []
        params = {}
        
        if condition_type == "low_sodium":
            conditions.append("sodium_mg < :max_sodium")
            params['max_sodium'] = kwargs.get('max_sodium', 200)
            
        elif condition_type == "high_protein":
            conditions.append("protein_g > :min_protein")
            params['min_protein'] = kwargs.get('min_protein', 10)
            
        elif condition_type == "low_fat":
            conditions.append("fat_g < :max_fat")
            params['max_fat'] = kwargs.get('max_fat', 5)
            
        elif condition_type == "diabetic_friendly":
            conditions.extend([
                "sugar_g < :max_sugar",
                "carbohydrate_g < :max_carb",
                "sodium_mg < :max_sodium"
            ])
            params.update({
                'max_sugar': kwargs.get('max_sugar', 5),
                'max_carb': kwargs.get('max_carb', 30),
                'max_sodium': kwargs.get('max_sodium', 300)
            })
        
        if conditions:
            query = base_query + " AND " + " AND ".join(conditions) + " ORDER BY RAND() LIMIT 50"
        else:
            query = base_query + " ORDER BY RAND() LIMIT 50"
        
        df = pd.read_sql(text(query), engine, params=params)
        return df
        
    except Exception as e:
        print(f"조건별 음식 검색 오류: {e}")
        return pd.DataFrame()

# ---------- PDF 파일 로딩 (기존 유지) ----------
pdf_loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_documents = pdf_loader.load()

print(f"로드된 PDF 문서: {len(pdf_documents)}개")

# 텍스트 분할
pdf_text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separator="\n"
)
nutrition_text_splitter = CharacterTextSplitter(
    chunk_size=200,  # 영양 정보는 더 작은 청크로
    chunk_overlap=20,
    separator="\n"
)

pdf_docs = pdf_text_splitter.split_documents(pdf_documents) if pdf_documents else []

# MySQL에서 영양 데이터 가져오기
nutrition_docs = fetch_nutrition_data_from_mysql()
if nutrition_docs:
    nutrition_docs = nutrition_text_splitter.split_documents(nutrition_docs)

print(f"분할된 PDF 청크: {len(pdf_docs)}개")
print(f"분할된 영양정보 청크: {len(nutrition_docs)}개")

# ---------- 벡터 DB ----------
embedding = OpenAIEmbeddings()

import shutil
if os.path.exists("./vector_db_pdf"):
    shutil.rmtree("./vector_db_pdf")
if os.path.exists("./vector_db_nutrition"):
    shutil.rmtree("./vector_db_nutrition")

if pdf_docs:
    vectordb_pdf = Chroma.from_documents(pdf_docs, embedding, persist_directory="./vector_db_pdf")
    vectordb_pdf.persist()
    retriever_pdf = vectordb_pdf.as_retriever(search_kwargs={"k": 10})
else:
    vectordb_pdf = None
    retriever_pdf = None

if nutrition_docs:
    vectordb_nutrition = Chroma.from_documents(nutrition_docs, embedding, persist_directory="./vector_db_nutrition")
    vectordb_nutrition.persist()
    retriever_nutrition = vectordb_nutrition.as_retriever(search_kwargs={"k": 15})
else:
    vectordb_nutrition = None
    retriever_nutrition = None

# ---------- LLM 및 Chain 설정 ----------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=500)

# PDF용 QA Chain (의료 정보) - 기존과 동일
if retriever_pdf:
    pdf_prompt = PromptTemplate(
        template="""**중요: 아래 제공된 문서 내용만을 사용하여 답변하세요. 문서가 영어인 경우 한국어로 번역하여 답변하세요.**

제공된 문서:
{context}

질문: {question}

답변 규칙:
1. 위 문서에 명시된 내용만 사용
2. 영어 문서의 경우 한국어로 번역하여 답변
3. 문서에 직접적으로 명시되어 있지 않더라도 관련 정보가 있으면 간접적으로 설명해도 됩니다. 단, 문서에 기반한 추론임을 명시하세요.
4. 의료적 정보는 정확하게 전달하되, "의사와 상담" 같은 멘트는 절대 포함하지 마세요

답변:""",
        input_variables=["context", "question"]
    )
    qa_chain_pdf = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever_pdf, 
        chain_type="stuff",
        chain_type_kwargs={"prompt": pdf_prompt}
    )

# 영양정보용 QA Chain (MySQL 데이터)
if retriever_nutrition:
    nutrition_prompt = PromptTemplate(
        template="""**중요: 오직 아래 제공된 영양 데이터만을 사용하여 답변하세요.**

제공된 영양 데이터:
{context}

질문: {question}

답변 규칙:
1. 위 데이터에 있는 음식과 영양성분만 언급
2. 데이터에 없는 음식은 언급하지 마세요
3. 데이터에 없으면 "제공된 영양 데이터에서 해당 정보를 찾을 수 없습니다"라고 답변

답변:""",
        input_variables=["context", "question"]
    )
    qa_chain_nutrition = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever_nutrition, 
        chain_type="stuff",
        chain_type_kwargs={"prompt": nutrition_prompt}
    )

# ---------- Tool 함수들 ----------
def extract_disease_tool_func(question: str) -> str:
    extract_prompt = PromptTemplate(
        input_variables=["question"],
        template="""질문에서 질병명이나 건강 상태만 간단히 추출하세요. 영어든 한국어든 상관없이 추출하세요.

예시:
질문: "당뇨가 있는데 저녁 추천해줘" → 답변: "당뇨"
질문: "고혈압 환자인데 뭘 먹을까?" → 답변: "고혈압"
질문: "나는 gout 가 있는데 저녁 메뉴 추천 해주셈" → 답변: "gout"
질문: "나는 간경변이 있는데 저녁 메뉴 추천 해주셈" → 답변: "간경변"

질문: {question}
답변:"""
    )
    from langchain.chains import LLMChain
    extract_chain = LLMChain(llm=llm, prompt=extract_prompt)
    result = extract_chain.run({"question": question}).strip()
    print(f"\n추출된 질병명: {result}")
    return result

def get_avoid_foods_tool_func(disease: str) -> str:
    if not retriever_pdf:
        return "PDF 문서가 없어 피해야 할 음식 정보를 찾을 수 없습니다."
    
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
        result = qa_chain_pdf.run(search_query)
        if "제공된 문서에서" not in result and len(result.strip()) > 20:
            return f"{disease} 환자가 피해야 할 음식 정보:\n{result}"
        else:
            return f"제공된 문서에서 {disease}의 피해야 할 음식 정보를 찾을 수 없습니다."
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"

def get_nutrition_standards_tool_func(disease: str) -> str:
    if not retriever_pdf:
        return "PDF 문서가 없어 영양 기준 정보를 찾을 수 없습니다."
    
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
        result = qa_chain_pdf.run(search_query)
        if "제공된 문서에서" not in result and len(result.strip()) > 20:
            return f"{disease} 환자의 영양 기준 정보:\n{result}"
        else:
            return f"제공된 문서에서 {disease}의 영양 기준 정보를 찾을 수 없습니다."
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"

def recommend_foods_tool_func(input_str: str) -> str:
    """
    입력 형식: "질병명|피해야할음식|영양기준"
    MySQL 데이터베이스에서 조건에 맞는 음식 추천
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
    
    print(f"\n음식 추천 - 질병: {disease}, 피해야할음식: {avoid_foods}")
    
    # 질병별 맞춤 MySQL 검색
    try:
        condition_foods = pd.DataFrame()
        disease_lower = disease.lower()
        
        if '당뇨' in disease or 'diabetes' in disease_lower:
            condition_foods = search_foods_by_condition('diabetic_friendly')
        elif '고혈압' in disease or 'hypertension' in disease_lower:
            condition_foods = search_foods_by_condition('low_sodium', max_sodium=150)
        elif '고지혈증' in disease or 'hyperlipidemia' in disease_lower:
            condition_foods = search_foods_by_condition('low_fat', max_fat=3)
        else:
            # 일반적인 건강식 조건
            condition_foods = search_foods_by_condition('high_protein', min_protein=8)
        
        if condition_foods.empty:
            # Vector DB 검색으로 대체
            health_query = f"{disease} 건강식 저염분 고단백"
            docs = retriever_nutrition.get_relevant_documents(health_query)
            context = "\n".join([doc.page_content for doc in docs])
        else:
            # MySQL 결과를 텍스트로 변환
            context = f"질병: {disease}\n피해야 할 음식: {avoid_foods}\n영양 기준: {nutrition_standards}\n\n"
            context += "추천 가능한 음식들:\n"
            
            for _, row in condition_foods.head(10).iterrows():
                food_info = f"음식이름: {row['name']} | "
                food_info += f"에너지: {row['energy_kcal']:.1f}kcal | "
                food_info += f"단백질: {row['protein_g']:.1f}g | "
                food_info += f"지방: {row['fat_g']:.1f}g | "
                food_info += f"탄수화물: {row['carbohydrate_g']:.1f}g | "
                food_info += f"나트륨: {int(row['sodium_mg'])}mg"
                context += food_info + "\n"
        
        # 추천 생성
        recommendation_prompt = PromptTemplate(
            template="""다음 정보를 바탕으로 {disease} 환자에게 적합한 저녁 음식 3가지를 추천하세요.

{context}

추천 규칙:
1. 피해야 할 음식에 포함된 것은 절대 추천하지 마세요
2. 영양 기준에 맞는 음식만 선택하세요
3. 각 음식의 구체적인 영양성분을 명시하세요
4. {disease} 환자에게 왜 좋은지 간단히 설명하세요
5. 제공된 데이터에 있는 음식만 추천하세요

추천 음식:""",
            input_variables=["disease", "context"]
        )
        
        from langchain.chains import LLMChain
        recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)
        result = recommendation_chain.run({
            "disease": disease,
            "context": context
        })
        
        return result
        
    except Exception as e:
        print(f"음식 추천 오류: {e}")
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
        description="특정 질병 환자가 피해야 할 음식 정보를 PDF 문서에서 검색합니다."
    ),
    Tool(
        name="GetNutritionStandards",
        func=get_nutrition_standards_tool_func, 
        description="특정 질병 환자의 영양 섭취 기준을 PDF 문서에서 검색합니다."
    ),
    Tool(
        name="RecommendFoods",
        func=recommend_foods_tool_func,
        description="MySQL 데이터베이스에서 질병 조건에 맞는 적합한 음식을 추천합니다."
    )
]

# ---------- API 모델 ----------
class AskRequest(BaseModel):
    question: str

class DebugRequest(BaseModel):
    query: str
    doc_type: str  # 'pdf' or 'nutrition'

@app.get("/")
def root():
    return {
        "message": "MySQL 연동 RAG 기반 식단 추천 API 서버 실행 중",
        "pdf_docs": len(pdf_docs) if pdf_docs else 0,
        "nutrition_docs": len(nutrition_docs) if nutrition_docs else 0,
        "mysql_connection": "연결됨" if engine else "연결 안됨"
    }

@app.get("/test-mysql")
def test_mysql():
    """MySQL 연결 테스트"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM food_nutritional_ingredients"))
            count = result.fetchone()[0]
            return {"status": "success", "total_foods": count}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/debug")
def debug_search(request: DebugRequest):
    """문서 검색 디버깅용 엔드포인트"""
    try:
        if request.doc_type == 'pdf' and retriever_pdf:
            docs = retriever_pdf.get_relevant_documents(request.query)
            return {
                "query": request.query,
                "doc_type": request.doc_type,
                "found_docs": len(docs),
                "content": [doc.page_content[:500] for doc in docs]
            }
        elif request.doc_type == 'nutrition' and retriever_nutrition:
            docs = retriever_nutrition.get_relevant_documents(request.query)
            return {
                "query": request.query,
                "doc_type": request.doc_type, 
                "found_docs": len(docs),
                "content": [doc.page_content[:200] for doc in docs]
            }
        else:
            return {"error": f"{request.doc_type} 문서가 로드되지 않았습니다."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
def ask(request: AskRequest):
    try:
        print(f"받은 질문: {request.question}")
        
        # Agent 초기화
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        enhanced_question = f"""
        사용자 질문: {request.question}
        
        **다음 단계를 순서대로 수행하세요:**
        
        1. ExtractDisease로 질병명 추출
        2. GetAvoidFoods로 해당 질병에서 피해야 할 음식 확인
        3. GetNutritionStandards로 해당 질병의 영양 기준 확인
        4. RecommendFoods에 "질병명|피해야할음식|영양기준" 형태로 전달하여 MySQL 데이터베이스에서 적합한 음식 추천
        
        **중요 규칙:**
        - PDF 문서가 영어인 경우 한국어로 번역하여 설명
        - MySQL 데이터베이스와 Vector DB 검색 결과에만 기반하여 답변
        - "의사와 상담" 같은 멘트는 절대 포함하지 마세요
        - 4단계 모두 완료한 후 종합적인 최종 답변 작성
        """
        
        result = agent.run(enhanced_question)
        return {"response": result}
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)