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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
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

# 전역 벡터스토어 및 검색기 관리 객체 (간이 세션 저장소)
vector_store = None
retriever = None

# ---------------------------------------------------------
# 1. 실시간 스트리밍 핸들러 (StreamHandler) 정의
# ---------------------------------------------------------
class StreamHandler(BaseCallbackHandler):
    """LLM이 생성하는 토큰을 실시간으로 가로채어 스트리밍 형태로 전달하기 위한 핸들러"""
    def __init__(self, queue: AsyncGenerator):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """새로운 토큰이 생성될 때마다 호출됨"""
        # 실제 프로덕션 환경에서는 asyncio.Queue 등을 활용하여 토큰을 비동기 전달
        pass


# ---------------------------------------------------------
# 요청 모델 정의
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


# ---------------------------------------------------------
# 2. PDF 처리 및 Vector DB 생성 엔드포인트
# ---------------------------------------------------------
@app.post("/api/vectorstore/create")
async def create_vector_store(file: UploadFile = File(...)):
    """업로드된 PDF 파일을 읽어 텍스트를 추출하고, 청크로 분할한 뒤 Chroma DB에 적재합니다."""
    global vector_store, retriever
    
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # [수정] 텍스트 청크 크기를 키우고 오버랩을 줄여서 전체 청크 개수 줄이기 (Timeout 방지)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY,
            # [추가] 대량 문서 처리 시 타임아웃 방지를 위한 배치 크기 제한 (지원하는 경우)
            batch_size=10
        )

        # Chroma 벡터스토어 생성 (청크가 너무 많으면 이 부분에서 504가 발생하므로 분할 처리 권장)
        vector_store = Chroma.from_documents(chunks, embeddings)

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

        return {
            "status": "success",
            "message": f"벡터 DB가 성공적으로 생성되었습니다. (총 청크 수: {len(chunks)}개)"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# ---------------------------------------------------------
# 3. 질의응답 및 스트리밍 답변 생성 엔드포인트
# ---------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    """사용자의 질의를 받아 MultiQueryRetriever로 문서를 검색하고, 현진건 작가 페르소나로 답변을 스트리밍합니다."""
    global retriever
    if not retriever:
        raise HTTPException(status_code=400, detail="Vector DB가 생성되지 않았습니다. 먼저 PDF를 업로드하여 DB를 생성하세요.")

    # 랭체인 프롬프트 허브 컨셉: 현진건 작가 페르소나 적용 프롬프트 템플릿
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

    # 텍스트 포맷팅 헬퍼 함수 (출처 정보 함께 구성)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # 검색된 문서 출처를 함께 반환하기 위한 체인 구성 (RunnablePassthrough 활용)
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
        # 먼저 검색된 출처 정보를 JSON 형태로 전송하거나 마크다운으로 구성
        import json
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
        
        # LLM 답변 본문 스트리밍 전송뮬
        chunk_size = 20
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i+chunk_size]
            yield f"data: {json.dumps({'type': 'content', 'data': chunk}, ensure_ascii=False)}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
