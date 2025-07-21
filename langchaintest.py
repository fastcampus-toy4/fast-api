import pandas as pd
import os
import time
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ---------- CSV -> TXT 변환 ----------
def convert_csv_to_text_optimized(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    sample_size = min(500, len(df))
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    column_mapping = {
        'foodNm': '음식이름', 'enerc': '에너지(kcal)', 'prot': '단백질(g)',
        'fatce': '지방(g)', 'chocdf': '탄수화물(g)', 'sugar': '당류(g)',
        'nat': '나트륨(mg)', 'chole': '콜레스테롤(mg)'
    }
    core_columns = list(column_mapping.keys())
    available_columns = [col for col in core_columns if col in df.columns]

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write("=== 영양 성분 데이터베이스 ===\n\n")
        for i, row in df.iterrows():
            items = []
            for col in available_columns:
                val = str(row[col]).strip()
                if pd.notna(val) and val and val != 'nan':
                    korean_col = column_mapping.get(col, col)
                    try:
                        numeric_val = float(val)
                        if numeric_val <= 0: continue
                        if col in ['enerc', 'prot', 'fatce', 'chocdf']:
                            val = f"{numeric_val:.1f}"
                        else:
                            val = f"{int(numeric_val)}"
                    except: 
                        pass
                    items.append(f"{korean_col}: {val}")
            
            if len(items) >= 3:
                food_info = " | ".join(items)
                f.write(f"{food_info}\n")

# ---------- 파일 로딩 ----------
csv_path = './docs/nutri_food_info.csv'
txt_path = './docs/nutri_food_info.txt'
if os.path.exists(csv_path):
    convert_csv_to_text_optimized(csv_path, txt_path)

pdf_loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
txt_loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=lambda path: TextLoader(path, encoding="utf-8-sig"))

pdf_documents = pdf_loader.load()
txt_documents = txt_loader.load()

print(f"로드된 PDF 문서: {len(pdf_documents)}개")
print(f"로드된 TXT 문서: {len(txt_documents)}개")

# 텍스트 분할
pdf_text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separator="\n"
)
txt_text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separator="\n"
)

pdf_docs = pdf_text_splitter.split_documents(pdf_documents) if pdf_documents else []
txt_docs = txt_text_splitter.split_documents(txt_documents) if txt_documents else []

print(f"분할된 PDF 청크: {len(pdf_docs)}개")
print(f"분할된 TXT 청크: {len(txt_docs)}개")

# ---------- 벡터 DB ----------
embedding = OpenAIEmbeddings()

import shutil
if os.path.exists("./vector_db_pdf"):
    shutil.rmtree("./vector_db_pdf")
if os.path.exists("./vector_db_txt"):
    shutil.rmtree("./vector_db_txt")

if pdf_docs:
    vectordb_pdf = Chroma.from_documents(pdf_docs, embedding, persist_directory="./vector_db_pdf")
    vectordb_pdf.persist()
    retriever_pdf = vectordb_pdf.as_retriever(search_kwargs={"k": 10})
else:
    vectordb_pdf = None
    retriever_pdf = None

if txt_docs:
    vectordb_txt = Chroma.from_documents(txt_docs, embedding, persist_directory="./vector_db_txt")
    vectordb_txt.persist()
    retriever_txt = vectordb_txt.as_retriever(search_kwargs={"k": 10})
else:
    vectordb_txt = None
    retriever_txt = None

# ---------- LLM 및 Chain 설정 ----------
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=500)

# PDF용 QA Chain (의료 정보)
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

# TXT용 QA Chain (영양 정보)
if retriever_txt:
    txt_prompt = PromptTemplate(
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
    qa_chain_txt = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever_txt, 
        chain_type="stuff",
        chain_type_kwargs={"prompt": txt_prompt}
    )

# ---------- 질병명 추출 프롬프트 ----------
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

# ---------- 검색 쿼리 생성 프롬프트 ----------
search_query_prompt = PromptTemplate(
    input_variables=["disease", "search_type"],
    template="""질병명 "{disease}"를 기반으로 {search_type} 정보를 찾기 위한 검색 쿼리를 생성하세요.

{search_type} 유형에 따른 가이드:
- avoid_foods: 피해야 할 음식, 금지 식품, 섭취 제한, 식이 조절, 나트륨/지방 제한, 고칼로리 피함 등 키워드를 포함
- nutrition_standards: 권장 섭취량, 식이요법, 영양 기준, 영양소 권장량, 식이지침, 에너지/단백질/지방/나트륨 조절 등 포함

다음을 포함하세요:
- 질병명 한국어 표현
- 질병명 영어 표현 (예: stroke, diabetes)
- 다양한 표현과 키워드를 OR 조건으로 나열
- 한국어와 영어 키워드를 모두 포함하여 다양한 표현으로 검색할 수 있도록 하세요.
- 영어 PDF 문서도 검색할 수 있도록 영어 의학 용어도 포함하세요.

예시 형식:
"stroke" OR "뇌졸중" AND ("nutrition guidelines" OR "식이요법" OR "저염식" OR ...)

질병명: {disease}
검색 타입: {search_type}

생성할 검색 쿼리:"""
)

# ---------- Tool 함수 개선 ----------
def extract_disease_tool_func(question: str) -> str:
    from langchain.chains import LLMChain
    extract_chain = LLMChain(llm=llm, prompt=extract_prompt)
    result = extract_chain.run({"question": question}).strip()
    print(f"\n추출된 질병명: {result}")
    return result

def get_avoid_foods_tool_func(disease: str) -> str:
    if not retriever_pdf:
        return "PDF 문서가 없어 피해야 할 음식 정보를 찾을 수 없습니다."
    
    from langchain.chains import LLMChain
    query_chain = LLMChain(llm=llm, prompt=search_query_prompt)
    
    # LLM을 사용해 검색 쿼리 생성
    search_query = query_chain.run({
        "disease": disease,
        "search_type": "avoid_foods"
    }).strip()
    
    print(f"\n생성된 검색 쿼리: {search_query}")
    
    try:
        result = qa_chain_pdf.run(search_query)
        docs = retriever_pdf.get_relevant_documents(search_query)
        print(f"검색 결과 - 문서 {len(docs)}개, 결과: {result[:100]}...")
        
        if "제공된 문서에서" not in result and len(result.strip()) > 20:
            return f"{disease} 환자가 피해야 할 음식 정보:\n{result}"
        else:
            return f"제공된 문서에서 {disease}의 피해야 할 음식 정보를 찾을 수 없습니다."
            
    except Exception as e:
        print(f"검색 오류: {e}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"

def get_nutrition_standards_tool_func(disease: str) -> str:
    if not retriever_pdf:
        return "PDF 문서가 없어 영양 기준 정보를 찾을 수 없습니다."
    
    from langchain.chains import LLMChain
    query_chain = LLMChain(llm=llm, prompt=search_query_prompt)
    
    # LLM을 사용해 검색 쿼리 생성
    search_query = query_chain.run({
        "disease": disease,
        "search_type": "nutrition_standards"
    }).strip()
    
    print(f"생성된 검색 쿼리: {search_query}")
    
    try:
        result = qa_chain_pdf.run(search_query)
        docs = retriever_pdf.get_relevant_documents(search_query)
        print(f"검색 결과 - 문서 {len(docs)}개, 결과: {result[:100]}...")
        
        if "제공된 문서에서" not in result and len(result.strip()) > 20:
            return f"{disease} 환자의 영양 기준 정보:\n{result}"
        else:
            return f"제공된 문서에서 {disease}의 영양 기준 정보를 찾을 수 없습니다."
            
    except Exception as e:
        print(f"검색 오류: {e}")
        return f"검색 중 오류가 발생했습니다: {str(e)}"

def recommend_foods_tool_func(input_str: str) -> str:
    """
    입력 형식: "질병명|피해야할음식|영양기준"
    """
    if not retriever_txt:
        return "영양 데이터가 없어 음식을 추천할 수 없습니다."
    
    # 입력 파싱
    parts = input_str.split("|")
    disease = parts[0] if len(parts) > 0 else ""
    avoid_foods = parts[1] if len(parts) > 1 else ""
    nutrition_standards = "|".join(parts[2:]) if len(parts) > 2 else ""
    
    if not disease:
        return "질병명이 제공되지 않아 추천이 불가능합니다."
    
    if not avoid_foods and not nutrition_standards:
        return f"{disease}에 대한 피해야 할 음식 정보나 영양 기준이 없으므로 추천할 수 없습니다."
    
    print(f"\n음식 추천 입력 파싱 - 질병: {disease}, 피해야할음식: {avoid_foods}")
    
    # 건강한 음식 검색을 위한 쿼리 생성
    health_query_prompt = PromptTemplate(
        template="""질병 "{disease}"에 적합한 건강한 음식을 영양 데이터베이스에서 찾기 위한 검색 키워드를 생성하세요.

다음과 같은 영양소와 건강 키워드를 포함하세요:
- 단백질, 비타민, 미네랄, 칼슘, 마그네슘 등 필수 영양소
- 저염분, 저나트륨, 고단백 등 건강 관련 키워드
- {disease}에 도움이 되는 영양소

질병: {disease}
검색 키워드:""",
        input_variables=["disease"]
    )
    
    from langchain.chains import LLMChain
    health_query_chain = LLMChain(llm=llm, prompt=health_query_prompt)
    query = health_query_chain.run({"disease": disease}).strip()
    
    print(f"영양 DB 검색 쿼리: {query}")
    
    # Vector DB에서 영양 데이터 검색
    docs = retriever_txt.get_relevant_documents(query)
    
    # 컨텍스트 구성
    context = f"""
    질병: {disease}
    피해야 할 음식: {avoid_foods}
    영양 기준: {nutrition_standards}
    
    다음 영양 데이터에서 위 조건에 맞는 음식만 추천하세요:
    """ + "\n".join([doc.page_content for doc in docs])
    
    # 맞춤형 프롬프트로 LLM 호출
    recommendation_prompt = PromptTemplate(
        template="""다음 정보를 바탕으로 {disease} 환자에게 적합한 저녁 음식 3가지를 추천하세요.

{context}

추천 규칙:
1. 피해야 할 음식에 포함된 것은 절대 추천하지 마세요
2. 영양 기준에 맞는 음식만 선택하세요
3. 각 음식의 구체적인 영양성분을 명시하세요
4. {disease} 환자에게 왜 좋은지 간단히 설명하세요
5. 제공된 영양 데이터에 있는 음식만 추천하세요
6. "의사와 상담" 같은 멘트는 절대 포함하지 마세요

추천 음식 (각 항목마다 이유 포함):

출력 형식 예시:
1. 음식명
   - 영양성분 요약:
   - 추천 이유:

2. 음식명
   - 영양성분 요약:
   - 추천 이유:

3. 음식명
   - 영양성분 요약:
   - 추천 이유:

추천 음식:""",
        input_variables=["disease", "context"]
    )
    
    recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)
    result = recommendation_chain.run({
        "disease": disease,
        "context": context
    })
    
    return result

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
        description="특정 질병 환자가 피해야 할 음식 정보를 PDF 문서에서 검색합니다. LLM이 자동으로 적절한 검색 쿼리를 생성합니다."
    ),
    Tool(
        name="GetNutritionStandards",
        func=get_nutrition_standards_tool_func, 
        description="특정 질병 환자의 영양 섭취 기준을 PDF 문서에서 검색합니다. LLM이 자동으로 적절한 검색 쿼리를 생성합니다."
    ),
    Tool(
        name="RecommendFoods",
        func=recommend_foods_tool_func,
        description="질병명|피해야할음식|영양기준 형태로 입력받아 영양 데이터베이스에서 적합한 음식을 추천합니다."
    )
]

# ---------- API 모델 ----------
class AskRequest(BaseModel):
    question: str

class DebugRequest(BaseModel):
    query: str
    doc_type: str  # 'pdf' or 'txt'

@app.get("/")
def root():
    return {
        "message": "RAG 기반 식단 추천 API 서버 실행 중",
        "pdf_docs": len(pdf_docs) if pdf_docs else 0,
        "txt_docs": len(txt_docs) if txt_docs else 0
    }

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
        elif request.doc_type == 'txt' and retriever_txt:
            docs = retriever_txt.get_relevant_documents(request.query)
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
        
        # 개선된 지시사항
        enhanced_question = f"""
        사용자 질문: {request.question}
        
        **다음 단계를 순서대로 수행하세요:**
        
        1. ExtractDisease로 질병명 추출
        2. GetAvoidFoods로 해당 질병에서 피해야 할 음식 확인 (LLM이 자동으로 적절한 검색 쿼리 생성)
        3. GetNutritionStandards로 해당 질병의 영양 기준 확인 (LLM이 자동으로 적절한 검색 쿼리 생성)
        4. RecommendFoods에 "질병명|피해야할음식|영양기준" 형태로 전달하여 적합한 음식 추천
        
        **중요 규칙:**
        - PDF 문서가 영어인 경우 한국어로 번역하여 설명
        - Vector DB 검색 결과에만 기반하여 답변
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