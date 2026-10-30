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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PERSIST_DIR = "/tmp/chroma_db"  # Cloud Run 컨테이너 쓰기 권한이 보장되는 /tmp 권장
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# API KEY 확인
api_key = os.environ.get("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)
# 1. 모델 변경: 경량/고속 모델인 gemini-2.0-flash-lite 적용
# max_tokens를 지정하여 응답 길이를 적정 수준으로 제한
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", 
    temperature=0.2,
    max_tokens=600,  # 원하는 최대 길이(글자/토큰)로 조절 가능
    google_api_key=api_key
)

# 2. 시스템 프롬프트: 핵심 위주로 간결하게 답변하도록 지침 추가
SYSTEM_PROMPT = """당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.

[답변 작성 규칙]
- 다음 컨텍스트를 참고하여 질문에 대해 장황한 서술을 피하고 핵심 내용 위주로 명확하고 간결하게 3~5줄 내외로 요약하여 답변하세요.

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
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )

        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {"status": "success", "message": f"{file.filename} 처리 및 벡터화 완료"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(req: QueryRequest):
    try:
        # Chroma 인스턴스 로드
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # 다중 질의 검색기 설정
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )

        # RAG 체인 파이프라인
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
        # 오류 내용을 명확하게 클라이언트로 전달
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")
