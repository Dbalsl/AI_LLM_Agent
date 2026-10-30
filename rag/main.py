import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# 모든 도메인에서의 CORS 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PERSIST_DIR = "/tmp/chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Google AI Studio API 키 확인
api_key = os.environ.get("GOOGLE_API_KEY")

# 1. 구글 임베딩 모델 설정
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key,
    request_options={"timeout": 120}
)

# 2. 고속 LLM 모델 설정 (gemini-2.0-flash-lite)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=api_key,
    temperature=0.1,
    max_tokens=500,
    request_options={"timeout": 60}
)

# 3. 시험 요구사항 시스템 프롬프트 정의
SYSTEM_PROMPT = """당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.

다음 컨텍스트를 참고하여 질문에 대해 장황하지 않게 핵심 위주로 명확하고 간결하게 3~5문장 내외로 답변하세요:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    """서버 상태 및 시험 요구사항 구성 정보 확인 엔드포인트"""
    return {
        "status": "connected",
        "vector_db": "Chroma DB",
        "embedding_model": "models/text-embedding-004",
        "llm_model": "gemini-2.0-flash-lite",
        "retriever": "MultiQueryRetriever",
        "query_extension": "RunnablePassthrough"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일 로드 및 Chroma DB에 분할 저장 (504 Timeout 방지 배치 분할)"""
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # PDF 파일 읽어오기
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # 텍스트 분할 (청크 수 조절)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Chroma DB 연결
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )

        # 구글 API 단일 요청 용량 초과(504) 방지를 위한 배치 적재
        batch_size = 25
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            vectorstore.add_documents(documents=batch)

        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "status": "success", 
            "message": f"'{file.filename}' 성공적으로 저장되었습니다. (총 {len(splits)}개 청크 색인 완료)"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(req: QueryRequest):
    """다중 질의 검색기 및 RunnablePassthrough를 적용한 RAG 파이프라인"""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 다중 질문 검색기 (MultiQueryRetriever)
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )

        # 질의 확장을 위한 RunnablePassthrough 파이프라인
        rag_chain = (
            {
                "context": multi_retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(req.question)
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
