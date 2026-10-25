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

# 프록시 환경 변수 자동 주입으로 인한 openai/httpx 충돌 방지
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

app = Flask(__name__)

# -------------------------------------------------------------
# 안전한 지연 초기화(Lazy Loading) 함수들
# (서버가 켜질 때가 아니라, 실제 기능이 실행될 때 객체를 생성하여 크래시 방지)
# -------------------------------------------------------------

def get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL 또는 SUPABASE_SERVICE_KEY가 설정되지 않았습니다.")
    return create_client(url, key)

def get_embeddings():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")

def get_vector_store():
    return SupabaseVectorStore(
        client=get_supabase_client(),
        embedding=get_embeddings(),
        table_name="documents",
        query_name="match_documents"
    )

def get_llm():
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=google_key,
        temperature=0.3,
        streaming=True
    )

@app.route("/health", methods=["GET"])
def health_check():
    """서버 상태 확인용 헬스체크 엔드포인트"""
    return jsonify({"status": "healthy", "message": "Cloud Run 서버가 정상 작동 중입니다."}), 200

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    """PDF 파일을 업로드받아 텍스트 추출 후 Supabase Vector DB에 저장"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "업로드된 PDF 파일이 없습니다."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "선택된 파일이 없습니다."}), 400

        filename = secure_filename(file.filename)
        file_bytes = file.read()
        
        # 1. PDF 텍스트 변환
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        raw_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
                
        if not raw_text.strip():
            return jsonify({"error": "PDF 파일에서 텍스트를 추출할 수 없습니다."}), 400

        # 2. 청크 단위 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"source": filename}]
        )

        # 3. Vector DB 저장
        v_store = get_vector_store()
        v_store.add_documents(chunks)

        return jsonify({
            "success": True, 
            "message": f"'{filename}' 파일이 성공적으로 처리되어 {len(chunks)}개의 청크가 벡터 DB에 저장되었습니다.",
            "chunk_count": len(chunks)
        }), 200

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"PDF 처리 중 서버 에러 발생: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """대화 기록 기반 질의 확장 및 RAG 스트리밍 답변 생성"""
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "질의 내용(query)이 누락되었습니다."}), 400

        user_query = data["query"]
        chat_history_raw = data.get("chat_history", [])

        chat_history = []
        for msg in chat_history_raw:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        current_llm = get_llm()

        # 1. Query Augmentation Chain 구성
        augmentation_prompt = ChatPromptTemplate.from_messages([
            ("system", "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        query_augmentation_chain = augmentation_prompt | current_llm | StrOutputParser()

        augmented_query = query_augmentation_chain.invoke({
            "chat_history": chat_history,
            "question": user_query
        })

        # 2. 벡터 DB 검색
        v_store = get_vector_store()
        retriever = v_store.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(augmented_query)
        
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        sources = list(set([doc.metadata.get("source", "알 수 없는 출처") for doc in retrieved_docs]))

        # 3. RAG 답변 생성
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야. 아래 제공된 참고 문서를 바탕으로 사용자의 질문에 정확하고 상세히 답변해줘. 만약 문서에 없는 내용이라면 모른다고 솔직하게 답변해."),
            ("human", "참고 문서:\n{context}\n\n사용자 질의: {augmented_query}")
        ])

        rag_chain = rag_prompt | current_llm | StrOutputParser()

        def generate():
            yield f"__SOURCES__:{json.dumps(sources, ensure_ascii=False)}\n"
            yield f"__AUGMENTED__:{augmented_query}\n"
            for chunk in rag_chain.stream({
                "context": context_text,
                "augmented_query": augmented_query
            }):
                yield chunk

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"답변 생성 중 서버 에러 발생: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
