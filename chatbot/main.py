import os
import shutil
from typing import AsyncGenerator
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# LangChain 및 구글 제나이 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler

# FastAPI 앱 초기화
app = FastAPI(title="현진건 작가 RAG 서버", version="1.0")

# ---------------------------------------------------------
# 환경 변수 설정 (GEMINI_API_KEY 확인)
# ---------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다.")

# 전역 벡터스토어, 검색기 및 백그라운드 작업 상태 관리 객체
vector_store = None
retriever = None
db_status = {"status": "idle", "message": "대기 중"}

# ---------------------------------------------------------
# 1. 실시간 스트리밍 핸들러 (StreamHandler) 정의
# ---------------------------------------------------------
class StreamHandler(BaseCallbackHandler):
    """LLM이 생성하는 토큰을 실시간으로 가로채어 스트리밍 형태로 전달하기 위한 핸들러"""
    def __init__(self, queue: AsyncGenerator):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """새로운 토큰이 생성될 때마다 호출됨"""
        pass


# ---------------------------------------------------------
# 요청 모델 정의
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


# ---------------------------------------------------------
# 2. 백그라운드 벡터 DB 생성 함수 (타임아웃 방지)
# ---------------------------------------------------------
def process_vector_store_background(file_path: str):
    """서버 타임아웃을 방지하기 위해 백그라운드에서 실행되는 PDF 파싱 및 임베딩 적재 로직"""
    global vector_store, retriever, db_status
    try:
        db_status = {"status": "processing", "message": "PDF 텍스트 추출 및 임베딩 생성 중..."}
        
        # PyPDFLoader를 사용하여 PDF 텍스트 추출
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # RecursiveCharacterTextSplitter를 활용하여 청크 단위로 분할 (크기 1000, 오버랩 100)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # 구글 임베딩 모델 설정 (models/text-embedding-004)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )

        # 타임아웃 방지를 위해 10개씩 배치 단위로 Chroma DB에 적재
        batch_size = 10
        temp_vector_store = None

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            if temp_vector_store is None:
                temp_vector_store = Chroma.from_documents(batch_chunks, embeddings)
            else:
                temp_vector_store.add_documents(batch_chunks)

        vector_store = temp_vector_store

        # 랭체인 다중 질문 검색기(MultiQueryRetriever) 구성
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm_for_query = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0, 
            google_api_key=GOOGLE_API_KEY
        )
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm_for_query
        )

        db_status = {"status": "completed", "message": f"벡터 DB 생성 완료! (총 청크 수: {len(chunks)}개)"}

    except Exception as e:
        db_status = {"status": "error", "message": f"오류 발생: {str(e)}"}
    finally:
        # 임시 파일 삭제 정리
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------------------------------
# 3. PDF 업로드 및 백그라운드 작업 등록 엔드포인트
# ---------------------------------------------------------
@app.post("/api/vectorstore/create")
async def create_vector_store(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """업로드된 PDF 파일을 받아 즉시 응답을 반환하고, 백그라운드에서 벡터 DB 생성을 수행합니다."""
    temp_file_path = f"temp_{file.filename}"
    
    # 업로드된 파일을 임시 저장
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 백그라운드 태스크 등록 (클라이언트 타임아웃 우회)
    background_tasks.add_task(process_vector_store_background, temp_file_path)

    return {
        "status": "accepted",
        "message": "PDF 파일이 접수되었습니다. 백그라운드에서 벡터 DB를 생성하고 있습니다."
    }


# ---------------------------------------------------------
# 4. 백그라운드 작업 상태 확인 엔드포인트
# ---------------------------------------------------------
@app.get("/api/vectorstore/status")
async def get_vector_store_status():
    """클라이언트가 벡터 DB 생성 진행 상황을 확인할 수 있는 상태 조회 API"""
    return db_status


# ---------------------------------------------------------
# 5. 질의응답 및 스트리밍 답변 생성 엔드포인트
# ---------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    """사용자의 질의를 받아 MultiQueryRetriever로 문서를 검색하고, 현진건 작가 페르소나로 답변을 스트리밍합니다."""
    global retriever
    if not retriever:
        raise HTTPException(status_code=400, detail="Vector DB가 아직 생성되지 않았거나 준비 중입니다. 잠시 후 다시 시도해 주세요.")

    # 현진건 작가 페르소나 적용 프롬프트 템플릿
    template = """당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. 
근대 한국 문학의 현실주의적 시각과 그 시대의 비극적 어조를 살려 사용자의 질문에 답변하세요.
제공된 참고 문서의 내용을 바탕으로 깊이 있는 문학적 답변을 구성해야 합니다.

[참고 문서]
{context}

[사용자 질문]
{question}

[현진건 작가의 답변]:"""

    prompt = ChatPromptTemplate.from_template(template)

    # 답변 생성 LLM 모델 설정 (gemini-1.5-flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
        streaming=True
    )

    # 텍스트 포맷팅 헬퍼 함수
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # 검색된 문서 출처를 함께 반환하기 위한 체인 구성
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # 검색 수행 및 문서 확보
    retrieved_docs = retriever.invoke(request.question)
    
    # 답변 생성 실행
    response_text = rag_chain_from_docs.invoke({
        "context": retrieved_docs,
        "question": request.question
    })

    # 출처 및 검색 내용 추출
    sources = [
        {"content": doc.page_content[:150] + "...", "source": doc.metadata.get("source", "알 수 없음"), "page": doc.metadata.get("page", 0) + 1}
        for doc in retrieved_docs
    ]

    # 클라이언트로 결과 및 출처를 스트리밍 형식으로 전송하기 위한 제너레이터
    async def generate():
        import json
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
        
        chunk_size = 20
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i+chunk_size]
            yield f"data: {json.dumps({'type': 'content', 'data': chunk}, ensure_ascii=False)}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
