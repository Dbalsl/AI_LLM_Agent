import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

# GAS 및 외부 연동을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PERSIST_DIR = "./chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# 1. 모델 및 임베딩 초기화 (환경 변수 GOOGLE_API_KEY 사용)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# 시스템 프롬프트 정의
SYSTEM_PROMPT = """당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.

다음 컨텍스트를 참고하여 서울시 도시 계획 전문가로서 전문적이고 명확하게 답변해 주세요:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {
        "status": "connected",
        "vector_db": "Chroma DB",
        "embedding_model": "models/text-embedding-004",
        "llm_model": "gemini-1.5-flash",
        "retriever": "MultiQueryRetriever",
        "query_extension": "RunnablePassthrough"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 문서를 수신하여 텍스트 분할 후 Chroma DB에 벡터화 저장"""
    try:
        temp_path = f"./temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        # Chroma DB 생성 및 임베딩 저장
        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )

        os.remove(temp_path)
        return {"status": "success", "message": f"{file.filename} 처리 및 벡터화 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(req: QueryRequest):
    """다중 질의 검색기 및 RunnablePassthrough를 적용한 질의응답 엔드포인트"""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # 다중 질문 검색기 (MultiQueryRetriever) 구성
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )

        # RAG 체인 구성 (RunnablePassthrough 적용)
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
        raise HTTPException(status_code=500, detail=str(e))
