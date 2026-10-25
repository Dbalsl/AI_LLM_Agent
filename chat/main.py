import os
import io
import json
from flask import Flask, request, Response, jsonify
from werkzeug.utils import secure_filename

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from supabase import create_client, Client

app = Flask(__name__)

# 환경 변수 설정
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Supabase 관련 환경 변수가 설정되지 않았습니다.")

# Supabase 클라이언트 초기화
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 임베딩 모델 (OpenAI 임베딩 사용)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# VectorStore 및 Retriever 설정
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# LLM 모델 설정 (gemini-3.5-flash-lite)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    streaming=True
)

@app.route("/health", methods=["GET"])
def health_check():
    """서버 상태 확인용 헬스체크 엔드포인트"""
    return jsonify({"status": "healthy", "message": "Cloud Run 서버가 정상 작동 중입니다."}), 200

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    """
    PDF 파일을 업로드받아 텍스트를 추출하고, 청크 단위로 쪼갠 뒤 
    OpenAI 임베딩으로 변환하여 Supabase Vector DB에 저장합니다.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "업로드된 PDF 파일이 없습니다."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "선택된 파일이 없습니다."}), 400

        filename = secure_filename(file.filename)
        file_bytes = file.read()
        
        # 1. PDF 텍스트 변환 (pypdf 사용)
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        raw_text = ""
        for page_idx, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
                
        if not raw_text.strip():
            return jsonify({"error": "PDF 파일에서 텍스트를 추출할 수 없습니다. (이미지 기반 PDF일 수 있음)"}), 400

        # 2. 청크 단위로 쪼개기 (RecursiveCharacterTextSplitter)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"source": filename}]
        )

        # 3. Vector DB에 청크 임베딩 저장
        vector_store.add_documents(chunks)

        return jsonify({
            "success": True, 
            "message": f"'{filename}' 파일이 성공적으로 처리되어 {len(chunks)}개의 청크가 벡터 DB에 저장되었습니다.",
            "chunk_count": len(chunks)
        }), 200

    except Exception as e:
        app.logger.error(f"PDF 업 및 처리 중 에러 발생: {str(e)}")
        return jsonify({"error": f"PDF 처리 중 서버 에러가 발생했습니다: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    사용자의 질문, 대화 기록을 받아 질의 확장을 수행하고, 
    Supabase RAG 검색 후 스트리밍 형식으로 답변과 출처를 반환합니다.
    """
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "질의 내용(query)이 누락되었습니다."}), 400

        user_query = data["query"]
        chat_history_raw = data.get("chat_history", []) # [{"role": "user"/"assistant", "content": "..."}]

        # 대화 기록 객체 변환
        chat_history = []
        for msg in chat_history_raw:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        # 1. Query Augmentation Chain 구성 (질의 확장)
        augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        query_augmentation_chain = augmentation_prompt | llm | StrOutputParser()

        # 2. 체인 실행 (데이터 주입 및 증강된 질의 생성)
        augmented_query = query_augmentation_chain.invoke({
            "chat_history": chat_history,
            "question": user_query
        })

        # 3. 증강된 질의로 벡터 DB 검색 (Retriever)
        retrieved_docs = retriever.invoke(augmented_query)
        
        # 검색된 문서 내용 및 출처 취합
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list(set([doc.metadata.get("source", "알 수 없는 출처") for doc in retrieved_docs]))

        # 4. RAG 답변 생성 프롬프트 구성
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 아래 제공된 참고 문서를 바탕으로 사용자의 질문에 정확하고 상세히 답변해줘. 만약 문서에 없는 내용이라면 모른다고 솔직하게 답변해."),
            ("human", "참고 문서:\n{context}\n\n사용자 질의: {augmented_query}")
        ])

        rag_chain = rag_prompt | llm | StrOutputParser()

        # 5. 실시간 스트리밍 응답 반환 (Server-Sent Events 또는 청크 스트리밍)
        def generate():
            # 먼저 출처 정보를 JSON 형태로 전송 후 데이터 스트리밍
            yield f"__SOURCES__:{json.dumps(sources, ensure_ascii=False)}\n"
            yield f"__AUGMENTED__:{augmented_query}\n"
            
            # LLM 스트리밍 응답 전달
            for chunk in rag_chain.stream({
                "context": context_text,
                "augmented_query": augmented_query
            }):
                yield chunk

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        app.logger.error(f"챗 처리 중 에러 발생: {str(e)}")
        return jsonify({"error": f"답변 생성 중 서버 에러가 발생했습니다: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
