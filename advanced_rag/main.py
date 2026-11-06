"""
================================================================================
[Advanced RAG Server - main.py]
FAISS Index + Multi-Query (5개 확장) + Hybrid Search (FAISS + BM25) + RAG Fusion (RRF)
================================================================================
- 프레임워크: FastAPI
- 주요 파이프라인:
  1) PDF 문서 로드 및 청킹 계층 (Document Ingestion & Chunking Layer)
  2) FAISS 벡터 인덱스(IndexFlatL2 / IndexHNSWFlat) 및 BM25 키워드 인덱스 구축 계층
  3) 다중 쿼리 확장 계층 (Multi-Query Layer: 단일 질의 -> 5가지 관점 파생 쿼리 생성)
  4) 하이브리드 검색 계층 (Hybrid Search Layer: FAISS Dense + BM25 Sparse 병행 검색)
  5) RRF 순위 융합 계층 (RAG Fusion & RRF Layer: 상호 순위 융합 및 Re-ranking)
  6) 실시간 스트리밍 핸들러 (StreamHandler: 단계별 진행 상태 및 LLM 답변 토큰 스트리밍)
================================================================================
"""

import os
import re
import io
import json
import time
import base64
import hashlib
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, AsyncGenerator

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Google GenAI SDK (선택적 로드: 환경변수 GEMINI_API_KEY가 있을 때 실제 LLM/임베딩 호출)
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ==============================================================================
# 1. FastAPI 앱 초기화 및 CORS 설정
# ==============================================================================
app = FastAPI(
    title="Advanced RAG Pipeline API (FAISS + Multi-Query + Hybrid + RRF)",
    description="FAISS 인덱스 기반 Hybrid Search, 5중 Multi-Query 확장, RAG Fusion(RRF) 백엔드 서버",
    version="2.0.0",
)

# GAS(Google Apps Script) 및 웹 클라이언트 요청 허용을 위한 CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 2. 데이터 모델 (Pydantic Schemas)
# ==============================================================================
class BuildDBRequest(BaseModel):
    """PDF 파일 기반 벡터 DB(FAISS + BM25) 구축 요청 모델"""
    file_name: str = Field(..., description="업로드된 PDF 파일명 (출처 표기용)")
    pdf_base64: Optional[str] = Field(None, description="Base64 인코딩된 PDF 바이너리 데이터")
    raw_text: Optional[str] = Field(None, description="PDF 텍스트 직접 전달 시 사용되는 대체 텍스트")
    index_type: str = Field(
        default="IndexFlatL2",
        description="FAISS 인덱스 종류 ('IndexFlatL2' 또는 'IndexHNSWFlat')"
    )
    chunk_size: int = Field(default=500, ge=100, le=2000, description="청크당 최대 글자 수")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="인접 청크 간 중첩 글자 수")
    api_key: Optional[str] = Field(None, description="선택적 Gemini API Key")


class QueryRequest(BaseModel):
    """사용자 질의 및 RAG Fusion 검색 요청 모델"""
    query: str = Field(..., min_length=1, description="사용자의 원본 질의")
    top_k: int = Field(default=5, ge=1, le=20, description="최종 반환할 상위 RRF 문서 개수")
    rrf_k: int = Field(default=60, ge=1, le=200, description="RRF 수식의 상수 k (기본값 60)")
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="FAISS Dense 검색 가중치")
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="BM25 Sparse 검색 가중치")
    stream: bool = Field(default=False, description="스트리밍 응답 여부")
    api_key: Optional[str] = Field(None, description="선택적 Gemini API Key")


# ==============================================================================
# 3. 실시간 스트리밍 핸들러 (StreamHandler)
# ==============================================================================
class StreamHandler:
    """
    [실시간 스트리밍 핸들러]
    파이프라인의 단계별 진행 상태(Status), 다중 쿼리 확장 결과, RRF 검색 문서,
    알고리즘 복잡도, LLM 생성 토큰을 NDJSON(Newline Delimited JSON) 포맷으로
    실시간 스트리밍하거나 버퍼링하여 클라이언트에 전달합니다.
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.start_time: float = time.perf_counter()

    def create_event(
        self,
        event_type: str,
        stage: str,
        data: Any,
        progress: int = 0,
        complexity: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        단일 스트리밍 이벤트 패킷(JSON 라인)을 생성합니다.
        - event_type: 'status' | 'multi_query' | 'retrieval' | 'token' | 'complete' | 'error'
        - stage: 현재 파이프라인 계층 이름
        - data: 전달할 페이로드
        - progress: 진행률 (0 ~ 100%)
        - complexity: 현재 단계의 시간/공간 복잡도 표기
        """
        elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        payload = {
            "event": event_type,
            "stage": stage,
            "progress": progress,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "complexity": complexity or {},
            "data": data,
        }
        self.events.append(payload)
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @staticmethod
    async def stream_answer_chunks(
        answer_text: str,
        handler: "StreamHandler",
        chunk_chars: int = 18,
    ) -> AsyncGenerator[str, None]:
        """
        생성된 LLM 답변을 작은 토큰 단위로 분할하여 실시간 스트리밍 전송합니다.
        """
        for i in range(0, len(answer_text), chunk_chars):
            token_piece = answer_text[i : i + chunk_chars]
            yield handler.create_event(
                event_type="token",
                stage="LLM Answer Generation Stream",
                data={"token": token_piece},
                progress=95,
                complexity={"time": "O(L_tokens)", "space": "O(L_tokens)"},
            )


# ==============================================================================
# 4. RAG 생성 진행 현황 상태 관리자 (Ingestion Progress Tracker)
# ==============================================================================
class IngestionProgressTracker:
    """
    서버에서 벡터 DB(FAISS + BM25)를 생성할 때 진행 현황(%)과 현재 단계,
    소요 시간, 에러 여부를 저장하여 클라이언트(GAS)가 실시간 조회할 수 있도록 관리합니다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.status: str = "IDLE"  # IDLE | PROCESSING | COMPLETED | ERROR
        self.progress: int = 0
        self.stage: str = "대기 중 (인덱스 미생성)"
        self.file_name: str = ""
        self.total_pages: int = 0
        self.total_chunks: int = 0
        self.index_type: str = "IndexFlatL2"
        self.detail_logs: List[str] = []
        self.error_info: Optional[Dict[str, Any]] = None
        self.updated_at: str = datetime.now(timezone.utc).isoformat()
        self.elapsed_ms: float = 0.0
        self._start_ts: float = time.perf_counter()

    def update(self, status: str, progress: int, stage: str, log_msg: Optional[str] = None):
        self.status = status
        self.progress = max(0, min(100, progress))
        self.stage = stage
        self.elapsed_ms = round((time.perf_counter() - self._start_ts) * 1000, 2)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if log_msg:
            self.detail_logs.append(f"[{self.elapsed_ms}ms] {log_msg}")

    def set_error(self, error_code: str, layer: str, message: str, resolution: str):
        self.status = "ERROR"
        self.stage = f"오류 발생 ({layer})"
        self.elapsed_ms = round((time.perf_counter() - self._start_ts) * 1000, 2)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.error_info = {
            "error_code": error_code,
            "layer": layer,
            "message": message,
            "resolution": resolution,
            "timestamp": self.updated_at,
        }
        self.detail_logs.append(f"[ERROR - {layer}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "file_name": self.file_name,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "index_type": self.index_type,
            "elapsed_ms": self.elapsed_ms,
            "detail_logs": self.detail_logs[-15:],
            "error_info": self.error_info,
            "updated_at": self.updated_at,
        }


# ==============================================================================
# 5. 핵심 RAG 엔진 (FAISS + Multi-Query + Hybrid Search + RRF Fusion)
# ==============================================================================
class AdvancedRAGEngine:
    """
    [FAISS + Multi-Query + Hybrid Search + RAG Fusion 통합 엔진]
    1) 임베딩 생성: Gemini Embedding (text-embedding-004) 또는 결정론적 고차원 N-gram 해시 임베딩(768차원)
    2) FAISS 인덱스: IndexFlatL2 (정확한 전수 L2 거리 계산) / IndexHNSWFlat (그래프 기반 초고속 근사 검색)
    3) Sparse 인덱스: BM25Okapi (한국어/영어 형태소·어절 및 바이그램 토큰 매칭)
    4) Multi-Query 확장: 단일 질의를 5가지 상호보완적 관점의 쿼리로 확장
    5) RRF (Reciprocal Rank Fusion): 다중 쿼리 × (FAISS + BM25) 순위를 상호 순위 융합 공식으로 재정렬
    """

    EMBEDDING_DIM = 768

    def __init__(self):
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.index_type: str = "IndexFlatL2"
        self.file_name: str = ""
        self.created_at: Optional[str] = None
        self.progress_tracker = IngestionProgressTracker()

    # --------------------------------------------------------------------------
    # [보조] 한국어/영문 하이브리드 토크나이저 (BM25 및 로컬 임베딩 공용)
    # --------------------------------------------------------------------------
    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """
        한국어 조사 분리 효과를 내기 위해 어절 단위 토큰과 2-음절(bi-gram) 토큰을 함께 추출합니다.
        시간 복잡도: O(L) (L: 문자열 길이)
        """
        cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
        words = [w for w in cleaned.split() if len(w) >= 1]
        tokens: List[str] = list(words)
        # 한국어 복합명사 및 조사 변화 대응을 위한 문자 바이그램(2-gram) 추가
        for word in words:
            if len(word) >= 2:
                for i in range(len(word) - 1):
                    tokens.append(word[i : i + 2])
        return tokens

    # --------------------------------------------------------------------------
    # [계층 1] 고차원 벡터 임베딩 생성 (Gemini API 또는 결정론적 768차원 폴백)
    # --------------------------------------------------------------------------
    def compute_embeddings(
        self, texts: List[str], api_key: Optional[str] = None
    ) -> np.ndarray:
        """
        문서/쿼리 리스트를 768차원 정규화 실수 벡터(float32) 행렬로 변환합니다.
        - GEMINI_API_KEY가 유효하면 Google `text-embedding-004` 모델을 사용합니다.
        - API 키가 없거나 오프라인 환경일 경우 의미적 어절/바이그램 해싱 기반 768차원 L2 정규화 벡터를 생성합니다.
        """
        effective_key = api_key or os.environ.get("GEMINI_API_KEY")
        if HAS_GENAI and effective_key:
            try:
                client = genai.Client(api_key=effective_key)
                vectors = []
                # 배치 처리로 임베딩 생성
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    resp = client.models.embed_content(
                        model="text-embedding-004",
                        contents=batch,
                    )
                    for emb in resp.embeddings:
                        vec = np.array(emb.values, dtype=np.float32)
                        if vec.shape[0] != self.EMBEDDING_DIM:
                            # 차원 불일치 시 패딩 또는 슬라이싱
                            padded = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
                            length = min(len(vec), self.EMBEDDING_DIM)
                            padded[:length] = vec[:length]
                            vec = padded
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec = vec / norm
                        vectors.append(vec)
                return np.vstack(vectors).astype(np.float32)
            except Exception:
                # API 호출 실패 시 로컬 결정론적 임베딩으로 안전하게 폴백
                pass

        # 결정론적 Feature Hashing 임베딩 (Signed Random Projection 방식, 768차원)
        matrix = np.zeros((len(texts), self.EMBEDDING_DIM), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = self.tokenize_text(text)
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                dim_idx = int.from_bytes(digest[:4], "little") % self.EMBEDDING_DIM
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                weight = 1.5 if len(token) > 2 else 1.0
                matrix[row_idx, dim_idx] += sign * weight
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix

    # --------------------------------------------------------------------------
    # [계층 2] PDF 텍스트 추출, 청킹 및 FAISS + BM25 인덱스 구축
    # --------------------------------------------------------------------------
    def build_vector_db(self, req: BuildDBRequest) -> Dict[str, Any]:
        """
        PDF 파일을 파싱하여 문맥 단위로 분할(Chunking)한 뒤,
        FAISS 고차원 벡터 인덱스(IndexFlatL2 또는 IndexHNSWFlat)와 BM25 인덱스를 동시에 구축합니다.
        """
        self.progress_tracker.reset()
        self.progress_tracker.file_name = req.file_name
        self.progress_tracker.index_type = req.index_type
        self.progress_tracker.update(
            status="PROCESSING",
            progress=10,
            stage="1/4단계: PDF 문서 디코딩 및 페이지별 텍스트 추출 중...",
            log_msg=f"파일 '{req.file_name}' 수신 완료. 파싱을 시작합니다.",
        )

        # 1. PDF 페이지별 텍스트 추출
        pages_data: List[Dict[str, Any]] = []
        if req.pdf_base64:
            try:
                # Data URL prefix(data:application/pdf;base64,) 제거 처리
                b64_clean = req.pdf_base64.split(",")[-1].strip()
                pdf_bytes = base64.b64decode(b64_clean)
                reader = PdfReader(io.BytesIO(pdf_bytes))
                for page_idx, page in enumerate(reader.pages):
                    extracted = (page.extract_text() or "").strip()
                    if extracted:
                        pages_data.append({"page": page_idx + 1, "text": extracted})
            except Exception as e:
                self.progress_tracker.set_error(
                    error_code="PDF_PARSE_ERROR",
                    layer="Document Ingestion Layer",
                    message=f"PDF 바이너리 파싱에 실패했습니다: {str(e)}",
                    resolution="손상되지 않은 표준 PDF 파일인지 확인하거나 텍스트가 포함된 PDF를 업로드해 주세요.",
                )
                raise ValueError(self.progress_tracker.error_info["message"])

        # raw_text가 함께 또는 단독으로 전달된 경우 처리
        if not pages_data and req.raw_text and req.raw_text.strip():
            paragraphs = [p.strip() for p in req.raw_text.split("\n\n") if p.strip()]
            for idx, para in enumerate(paragraphs):
                pages_data.append({"page": idx + 1, "text": para})

        if not pages_data:
            self.progress_tracker.set_error(
                error_code="EMPTY_PDF_TEXT",
                layer="Document Ingestion Layer",
                message="PDF에서 추출된 텍스트가 없습니다. (스캔 이미지 전용 PDF이거나 빈 문서입니다)",
                resolution="텍스트 선택이 가능한 PDF 파일을 선택하거나 OCR 처리된 문서를 사용해 주세요.",
            )
            raise ValueError(self.progress_tracker.error_info["message"])

        self.progress_tracker.total_pages = len(pages_data)
        self.progress_tracker.update(
            status="PROCESSING",
            progress=35,
            stage=f"2/4단계: 총 {len(pages_data)}페이지 문서 슬라이딩 윈도우 청킹(Chunking) 중...",
            log_msg=f"{len(pages_data)}개 페이지에서 텍스트 추출 성공. 청크 분할 수행 중.",
        )

        # 2. 슬라이딩 윈도우 문서 청킹 (메타데이터: 출처 파일명, 페이지 번호, 청크 ID 보존)
        chunks: List[Dict[str, Any]] = []
        chunk_id_counter = 1
        step = max(50, req.chunk_size - req.chunk_overlap)

        for page_item in pages_data:
            page_num = page_item["page"]
            text = re.sub(r"\s+", " ", page_item["text"]).strip()
            for start_idx in range(0, len(text), step):
                snippet = text[start_idx : start_idx + req.chunk_size].strip()
                if len(snippet) < 20 and chunks:
                    continue
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id_counter:04d}",
                        "source": req.file_name,
                        "page": page_num,
                        "text": snippet,
                        "char_length": len(snippet),
                    }
                )
                chunk_id_counter += 1

        self.progress_tracker.total_chunks = len(chunks)
        self.progress_tracker.update(
            status="PROCESSING",
            progress=65,
            stage=f"3/4단계: {len(chunks)}개 청크 고차원 벡터 임베딩(768차원) 변환 중...",
            log_msg=f"총 {len(chunks)}개 청크 생성 완료. 고차원 밀집 벡터 변환 시작.",
        )

        # 3. 임베딩 벡터 계산
        texts_to_embed = [c["text"] for c in chunks]
        embeddings = self.compute_embeddings(texts_to_embed, api_key=req.api_key)

        self.progress_tracker.update(
            status="PROCESSING",
            progress=85,
            stage=f"4/4단계: FAISS ({req.index_type}) 및 BM25 하이브리드 인덱스 빌드 중...",
            log_msg=f"임베딩 행렬 shape={embeddings.shape} 생성 완료. FAISS & BM25 인덱싱 수행.",
        )

        # 4. FAISS 인덱스 구축 (IndexFlatL2 vs IndexHNSWFlat)
        if req.index_type == "IndexHNSWFlat":
            # HNSW 그래프 인덱스: M=32(노드당 연결 수), efConstruction=64, efSearch=32
            index = faiss.IndexHNSWFlat(self.EMBEDDING_DIM, 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
            self.index_type = "IndexHNSWFlat"
        else:
            # 기본값: 정확한 L2 유클리드 거리 계산 인덱스
            index = faiss.IndexFlatL2(self.EMBEDDING_DIM)
            self.index_type = "IndexFlatL2"

        index.add(embeddings)
        self.faiss_index = index

        # 5. BM25 키워드(Sparse) 인덱스 구축
        self.tokenized_corpus = [self.tokenize_text(c["text"]) for c in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        self.documents = chunks
        self.file_name = req.file_name
        self.created_at = datetime.now(timezone.utc).isoformat()

        self.progress_tracker.update(
            status="COMPLETED",
            progress=100,
            stage="벡터 DB (FAISS + BM25) 구축 완료",
            log_msg=f"FAISS({self.index_type}) 및 BM25 인덱스 적재 완료! 총 {len(chunks)}개 청크 준비됨.",
        )

        return {
            "status": "SUCCESS",
            "file_name": self.file_name,
            "index_type": self.index_type,
            "embedding_dim": self.EMBEDDING_DIM,
            "total_pages": len(pages_data),
            "total_chunks": len(chunks),
            "created_at": self.created_at,
            "progress": self.progress_tracker.to_dict(),
            "complexity": {
                "chunking": "O(L_total_chars)",
                "faiss_build": (
                    "O(N * d)"
                    if self.index_type == "IndexFlatL2"
                    else "O(N * d * log N)"
                ),
                "bm25_build": "O(N * avg_tokens)",
                "space_complexity": "O(N * (d + avg_tokens))",
            },
        }

    # --------------------------------------------------------------------------
    # [계층 3] 다중 쿼리 확장 계층 (Multi-Query Layer: 5가지 관점 확장)
    # --------------------------------------------------------------------------
    def expand_multi_queries(
        self, original_query: str, api_key: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        사용자의 단일 입력 쿼리를 LLM(또는 규칙 기반 의미 확장기)을 통해
        서로 다른 5가지 관점의 파생 쿼리로 확장하여 단일 쿼리 검색의 어휘 불일치 및 누락을 방지합니다.
        시간 복잡도: O(M) (M = 5, 생성 쿼리 수)
        """
        perspectives = [
            "핵심 개념 및 정의 관점",
            "기술적 동작 원리 및 구조 관점",
            "실무 적용 및 구체적 사례 관점",
            "장단점·한계점 및 비교 분석 관점",
            "핵심 키워드 및 동의어 확장 관점",
        ]

        effective_key = api_key or os.environ.get("GEMINI_API_KEY")
        if HAS_GENAI and effective_key:
            try:
                client = genai.Client(api_key=effective_key)
                prompt = (
                    "당신은 RAG Fusion 검색 최적화 전문가입니다.\n"
                    f"사용자 원본 질의: \"{original_query}\"\n\n"
                    "위 단일 질의에 대해 검색 재현율(Recall)을 극대화할 수 있도록 다음 5가지 서로 다른 관점의 파생 검색 쿼리를 생성하세요.\n"
                    "1. 핵심 개념 및 정의 관점\n"
                    "2. 기술적 동작 원리 및 구조 관점\n"
                    "3. 실무 적용 및 구체적 사례 관점\n"
                    "4. 장단점·한계점 및 비교 분석 관점\n"
                    "5. 핵심 키워드 및 동의어 확장 관점\n\n"
                    "반드시 줄바꿈으로 구분된 5개의 한국어 검색 쿼리 문장만 출력하세요 (번호나 불릿 기호 제외)."
                )
                resp = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                lines = [
                    re.sub(r"^[\d\.\-\*\)\s]+", "", line).strip()
                    for line in (resp.text or "").splitlines()
                    if line.strip()
                ]
                if len(lines) >= 5:
                    return [
                        {
                            "id": f"MQ-{idx + 1}",
                            "perspective": perspectives[idx],
                            "query": lines[idx],
                        }
                        for idx in range(5)
                    ]
            except Exception:
                pass

        # 결정론적 5대 관점 다중 쿼리 생성 (API 키가 없거나 네트워크 제한 시에도 즉각 동작)
        core_terms = " ".join(self.tokenize_text(original_query)[:6])
        fallback_queries = [
            f"{original_query}의 핵심 개념, 정의 및 기본 배경 설명",
            f"{original_query}의 세부 동작 원리, 내부 아키텍처 및 처리 절차",
            f"{original_query}의 실제 활용 사례, 적용 방법 및 구현 가이드",
            f"{original_query}의 주요 특징, 장단점, 한계점 및 주의사항 분석",
            f"{original_query} 관련 주요 키워드 요약 ({core_terms})",
        ]
        return [
            {
                "id": f"MQ-{idx + 1}",
                "perspective": perspectives[idx],
                "query": fallback_queries[idx],
            }
            for idx in range(5)
        ]

    # --------------------------------------------------------------------------
    # [계층 4 & 5] 하이브리드 검색(FAISS + BM25) 및 RRF 순위 융합 계층
    # --------------------------------------------------------------------------
    def hybrid_search_with_rrf(
        self,
        original_query: str,
        expanded_queries: List[Dict[str, str]],
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        r"""
        확장된 5개의 쿼리(및 원본 쿼리) 각각에 대해:
        1) FAISS Dense Vector 검색 (의미적 유사도, L2 거리 오름차순)
        2) BM25 Sparse Keyword 검색 (정확한 단어/형태소 일치율, 점수 내림차순)
        을 병행 수행한 뒤, Reciprocal Rank Fusion(RRF) 공식으로 통합 가중치 점수를 산출하여
        가장 관련성이 높은 상위 K개 문서를 재정렬(Re-ranking)합니다.

        [RRF 수식]
        RRF_Score(d) = \sum_{q \in Q} ( w_dense / (k + rank_dense(d, q)) + w_sparse / (k + rank_sparse(d, q)) )
        """
        if self.faiss_index is None or self.bm25_index is None or not self.documents:
            raise ValueError("벡터 DB가 아직 생성되지 않았습니다. 먼저 PDF 파일을 선택하고 '벡터 DB 생성'을 실행해 주세요.")

        num_docs = len(self.documents)
        candidate_k = min(num_docs, max(top_k * 3, 15))

        # 원본 질의 + 5가지 확장 쿼리 모두 검색 경로에 투입
        all_queries = [
            {"id": "ORIG", "perspective": "사용자 원본 질의", "query": original_query}
        ] + expanded_queries

        # 모든 쿼리를 한 번에 배치 임베딩하여 속도 최적화
        query_Strings = [q["query"] for q in all_queries]
        query_embeddings = self.compute_embeddings(query_Strings, api_key=api_key)

        # FAISS 일괄 검색: distances shape = (M, candidate_k), indices shape = (M, candidate_k)
        faiss_distances, faiss_indices = self.faiss_index.search(
            query_embeddings, candidate_k
        )

        # 문서별 RRF 누적 점수 및 상세 추적 테이블
        rrf_records: Dict[int, Dict[str, Any]] = {}

        for q_idx, q_info in enumerate(all_queries):
            q_id = q_info["id"]
            q_text = q_info["query"]

            # (A) FAISS Dense 검색 순위 반영
            for rank_zero, doc_idx in enumerate(faiss_indices[q_idx]):
                if doc_idx < 0 or doc_idx >= num_docs:
                    continue
                rank = rank_zero + 1  # 1-based rank
                l2_dist = float(faiss_distances[q_idx][rank_zero])
                # L2 거리를 0~1 유사도로 변환: sim = 1 / (1 + d)
                dense_sim = round(1.0 / (1.0 + max(0.0, l2_dist)), 4)
                rrf_contrib = dense_weight * (1.0 / (rrf_k + rank))

                rec = rrf_records.setdefault(
                    int(doc_idx),
                    {
                        "doc_index": int(doc_idx),
                        "rrf_score": 0.0,
                        "dense_rrf_score": 0.0,
                        "sparse_rrf_score": 0.0,
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "best_dense_similarity": 0.0,
                        "best_bm25_score": 0.0,
                        "matched_queries": set(),
                        "rank_trace": [],
                    },
                )
                rec["rrf_score"] += rrf_contrib
                rec["dense_rrf_score"] += rrf_contrib
                if rec["best_dense_rank"] is None or rank < rec["best_dense_rank"]:
                    rec["best_dense_rank"] = rank
                if dense_sim > rec["best_dense_similarity"]:
                    rec["best_dense_similarity"] = dense_sim
                rec["matched_queries"].add(q_id)
                rec["rank_trace"].append(
                    f"{q_id}[FAISS #{rank} (sim:{dense_sim})]"
                )

            # (B) BM25 Sparse 키워드 검색 순위 반영
            q_tokens = self.tokenize_text(q_text)
            bm25_scores = self.bm25_index.get_scores(q_tokens)
            ranked_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

            for rank_zero, doc_idx in enumerate(ranked_bm25_indices):
                raw_bm25 = float(bm25_scores[doc_idx])
                if raw_bm25 <= 0.0 and rank_zero >= top_k:
                    continue
                rank = rank_zero + 1
                rrf_contrib = sparse_weight * (1.0 / (rrf_k + rank))

                rec = rrf_records.setdefault(
                    int(doc_idx),
                    {
                        "doc_index": int(doc_idx),
                        "rrf_score": 0.0,
                        "dense_rrf_score": 0.0,
                        "sparse_rrf_score": 0.0,
                        "best_dense_rank": None,
                        "best_sparse_rank": None,
                        "best_dense_similarity": 0.0,
                        "best_bm25_score": 0.0,
                        "matched_queries": set(),
                        "rank_trace": [],
                    },
                )
                rec["rrf_score"] += rrf_contrib
                rec["sparse_rrf_score"] += rrf_contrib
                if rec["best_sparse_rank"] is None or rank < rec["best_sparse_rank"]:
                    rec["best_sparse_rank"] = rank
                if raw_bm25 > rec["best_bm25_score"]:
                    rec["best_bm25_score"] = round(raw_bm25, 4)
                rec["matched_queries"].add(q_id)
                rec["rank_trace"].append(
                    f"{q_id}[BM25 #{rank} (score:{round(raw_bm25, 2)})]"
                )

        # RRF 통합 점수 내림차순으로 최종 Re-ranking 수행
        sorted_records = sorted(
            rrf_records.values(), key=lambda x: x["rrf_score"], reverse=True
        )[:top_k]

        reranked_documents: List[Dict[str, Any]] = []
        for final_rank, rec in enumerate(sorted_records, start=1):
            doc_meta = self.documents[rec["doc_index"]]
            reranked_documents.append(
                {
                    "final_rank": final_rank,
                    "chunk_id": doc_meta["chunk_id"],
                    "source": doc_meta["source"],
                    "page": doc_meta["page"],
                    "snippet": doc_meta["text"],
                    "char_length": doc_meta["char_length"],
                    "rrf_score": round(rec["rrf_score"], 6),
                    "dense_rrf_score": round(rec["dense_rrf_score"], 6),
                    "sparse_rrf_score": round(rec["sparse_rrf_score"], 6),
                    "best_dense_rank": rec["best_dense_rank"],
                    "best_sparse_rank": rec["best_sparse_rank"],
                    "best_dense_similarity": rec["best_dense_similarity"],
                    "best_bm25_score": rec["best_bm25_score"],
                    "matched_query_count": len(rec["matched_queries"]),
                    "matched_queries": sorted(list(rec["matched_queries"])),
                    "rank_trace_summary": ", ".join(rec["rank_trace"][:6]),
                }
            )

        return {
            "total_candidates_evaluated": len(rrf_records),
            "reranked_documents": reranked_documents,
        }

    # --------------------------------------------------------------------------
    # [계층 6] RRF 상위 문서 기반 LLM 최종 답변 생성 (출처 인용 포함)
    # --------------------------------------------------------------------------
    def generate_answer(
        self,
        query: str,
        reranked_docs: List[Dict[str, Any]],
        api_key: Optional[str] = None,
    ) -> str:
        """
        RRF로 재정렬된 상위 문서 스니펫들을 컨텍스트로 주입하여 출처(파일명, 페이지, 청크ID)가
        명시된 최종 답변을 생성합니다.
        """
        if not reranked_docs:
            return "검색된 관련 문서가 없어 답변을 생성할 수 없습니다."

        context_blocks = []
        for doc in reranked_docs:
            context_blocks.append(
                f"[출처 #{doc['final_rank']}: {doc['source']} | p.{doc['page']} | {doc['chunk_id']} | RRF={doc['rrf_score']}]\n"
                f"{doc['snippet']}"
            )
        joined_context = "\n\n".join(context_blocks)

        effective_key = api_key or os.environ.get("GEMINI_API_KEY")
        if HAS_GENAI and effective_key:
            try:
                client = genai.Client(api_key=effective_key)
                prompt = (
                    "당신은 정확하고 신뢰할 수 있는 Advanced RAG AI 어시스턴트입니다.\n"
                    "아래 제공된 [RRF 상위 검색 문서]의 내용만을 근거로 사용자의 질문에 체계적이고 상세하게 한국어로 답변하세요.\n"
                    "반드시 문장 끝에 사용한 근거의 출처 표기(예: [출처 #1: 파일명, p.페이지])를 명시하세요.\n\n"
                    f"=== [RRF 상위 검색 문서] ===\n{joined_context}\n\n"
                    f"=== [사용자 질문] ===\n{query}\n"
                )
                resp = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                if resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        # API 키 미설정 시에도 RRF 검색 결과를 구조화하여 출처와 함께 명확한 종합 답변 제공
        top_doc = reranked_docs[0]
        summary_lines = [
            f"질문하신 **\"{query}\"**에 대해 Multi-Query(5가지 관점 확장) 및 Hybrid Search(FAISS + BM25) 후 RRF(상호 순위 융합)로 도출된 핵심 답변입니다.\n",
            f"### 1. 핵심 요약 (최상위 근거 중심)",
            f"{top_doc['snippet']} "
            f"**[출처 #{top_doc['final_rank']}: {top_doc['source']}, p.{top_doc['page']}, {top_doc['chunk_id']}, RRF 점수: {top_doc['rrf_score']}]**\n",
        ]

        if len(reranked_docs) > 1:
            summary_lines.append("### 2. 보충 근거 및 다각도 검색 종합")
            for doc in reranked_docs[1:]:
                preview = doc["snippet"][:220] + ("..." if len(doc["snippet"]) > 220 else "")
                summary_lines.append(
                    f"- **근거 #{doc['final_rank']}** (FAISS 최고순위: #{doc['best_dense_rank'] or '-'}, "
                    f"BM25 최고순위: #{doc['best_sparse_rank'] or '-'}, RRF: `{doc['rrf_score']}`): "
                    f"{preview} **[출처: {doc['source']}, p.{doc['page']}, {doc['chunk_id']}]**"
                )

        return "\n".join(summary_lines)


# 싱글톤 엔진 인스턴스 생성
rag_engine = AdvancedRAGEngine()


# ==============================================================================
# 6. API 엔드포인트 정의
# ==============================================================================
@app.get("/health")
async def health_check():
    """
    서버 연동 상태, 현재 적재된 FAISS/BM25 인덱스 메타데이터, 알고리즘 복잡도 정보를 반환합니다.
    """
    num_docs = len(rag_engine.documents)
    is_ready = rag_engine.faiss_index is not None and num_docs > 0
    return {
        "status": "ONLINE",
        "server_time": datetime.now(timezone.utc).isoformat(),
        "vector_db_ready": is_ready,
        "file_name": rag_engine.file_name or None,
        "index_type": rag_engine.index_type,
        "total_chunks": num_docs,
        "embedding_dim": rag_engine.EMBEDDING_DIM,
        "algorithm_complexity": {
            "multi_query_expansion": {
                "time": "O(M) (M = 5 파생 쿼리)",
                "space": "O(M * L_query)",
                "description": "단일 질의를 5개의 상호보완적 관점 쿼리로 확장",
            },
            "faiss_dense_search": {
                "time": (
                    "O(M * N * d) [IndexFlatL2]"
                    if rag_engine.index_type == "IndexFlatL2"
                    else "O(M * d * log N) [IndexHNSWFlat]"
                ),
                "space": "O(N * d)",
                "description": "고차원 밀집 벡터(768차원) L2 유사도 검색",
            },
            "bm25_sparse_search": {
                "time": "O(M * |Q_tokens| * N)",
                "space": "O(N * V_vocab)",
                "description": "BM25Okapi 기반 키워드 및 어절/바이그램 역색인 매칭",
            },
            "rrf_rag_fusion": {
                "time": "O(M * K * log(M * K))",
                "space": "O(M * K)",
                "description": "Reciprocal Rank Fusion 수식 기반 다중 경로 순위 융합 및 재정렬",
            },
        },
    }


@app.get("/progress")
async def get_build_progress():
    """
    서버에서 진행 중인 RAG 벡터 DB 생성 현황(%) 및 단계별 로그를 반환합니다.
    """
    return rag_engine.progress_tracker.to_dict()


@app.post("/build-db")
async def build_vector_db_endpoint(req: BuildDBRequest):
    """
    PDF 파일(Base64 또는 텍스트)을 수신하여 FAISS + BM25 하이브리드 벡터 DB를 생성합니다.
    """
    try:
        result = rag_engine.build_vector_db(req)
        return result
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={
                "status": "ERROR",
                "error": rag_engine.progress_tracker.error_info
                or {
                    "error_code": "VALIDATION_ERROR",
                    "layer": "Document Ingestion Layer",
                    "message": str(ve),
                    "resolution": "요청 파라미터와 PDF 파일 내용을 확인해 주세요.",
                },
            },
        )
    except Exception as e:
        rag_engine.progress_tracker.set_error(
            error_code="INTERNAL_BUILD_ERROR",
            layer="FAISS & BM25 Indexing Layer",
            message=str(e),
            resolution="서버 로그를 확인하고 PDF 크기나 형식을 점검해 주세요.",
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error": rag_engine.progress_tracker.error_info,
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/query")
async def query_rag_pipeline(req: QueryRequest):
    """
    [메인 RAG Fusion 질의 엔드포인트]
    1) Multi-Query 5가지 확장
    2) FAISS (Dense) + BM25 (Sparse) 하이브리드 검색
    3) RRF (Reciprocal Rank Fusion) 재정렬
    4) LLM 답변 생성 및 단계별 스트리밍 이벤트 로그 반환
    """
    handler = StreamHandler()
    pipeline_start = time.perf_counter()

    try:
        # 1. 인덱스 준비 상태 검증
        if rag_engine.faiss_index is None or not rag_engine.documents:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "VECTOR_DB_NOT_INITIALIZED",
                    "layer": "Hybrid Search & FAISS Layer",
                    "message": "벡터 DB가 비어 있습니다. 먼저 PDF 파일을 선택하고 '벡터 DB 생성' 버튼을 눌러 인덱스를 구축해 주세요.",
                    "resolution": "좌측 패널에서 PDF 파일 요청 후 '벡터 DB 생성'을 완료한 뒤 질의해 주세요.",
                },
            )

        # 단계 1: Multi-Query 확장 (5가지 관점)
        t0 = time.perf_counter()
        handler.create_event(
            event_type="status",
            stage="Multi-Query Layer",
            data={"message": "단일 질의를 5가지 다각도 관점의 파생 쿼리로 확장 중..."},
            progress=20,
            complexity={"time": "O(M) (M=5)", "space": "O(M * L_query)"},
        )
        expanded_queries = rag_engine.expand_multi_queries(
            req.query, api_key=req.api_key
        )
        multi_query_ms = round((time.perf_counter() - t0) * 1000, 2)
        handler.create_event(
            event_type="multi_query",
            stage="Multi-Query Layer",
            data={"expanded_queries": expanded_queries, "latency_ms": multi_query_ms},
            progress=40,
            complexity={"time": "O(M) (M=5)", "space": "O(M * L_query)"},
        )

        # 단계 2 & 3: Hybrid Search (FAISS + BM25) 및 RRF 순위 융합
        t1 = time.perf_counter()
        faiss_time_str = (
            "O(M * N * d)"
            if rag_engine.index_type == "IndexFlatL2"
            else "O(M * d * log N)"
        )
        handler.create_event(
            event_type="status",
            stage="Hybrid Search & RRF Layer",
            data={
                "message": f"FAISS({rag_engine.index_type}) + BM25 병행 검색 및 RRF(k={req.rrf_k}) 재정렬 수행 중..."
            },
            progress=65,
            complexity={
                "time": f"Dense: {faiss_time_str} + Sparse: O(M * |Q| * N) + RRF: O(M * K * log K)",
                "space": "O(N * d + M * K)",
            },
        )
        fusion_result = rag_engine.hybrid_search_with_rrf(
            original_query=req.query,
            expanded_queries=expanded_queries,
            top_k=req.top_k,
            rrf_k=req.rrf_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
            api_key=req.api_key,
        )
        reranked_docs = fusion_result["reranked_documents"]
        hybrid_rrf_ms = round((time.perf_counter() - t1) * 1000, 2)

        handler.create_event(
            event_type="retrieval",
            stage="RAG Fusion & RRF Layer",
            data={
                "reranked_documents": reranked_docs,
                "total_candidates": fusion_result["total_candidates_evaluated"],
                "latency_ms": hybrid_rrf_ms,
            },
            progress=80,
            complexity={"time": "O(M * K * log(M * K))", "space": "O(M * K)"},
        )

        # 단계 4: LLM 최종 답변 생성
        t2 = time.perf_counter()
        answer = rag_engine.generate_answer(
            query=req.query,
            reranked_docs=reranked_docs,
            api_key=req.api_key,
        )
        llm_ms = round((time.perf_counter() - t2) * 1000, 2)
        total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

        # 스트리밍 토큰 이벤트 기록 (GAS StreamHandler 및 프론트엔드 재생 호환)
        async for _ in StreamHandler.stream_answer_chunks(answer, handler):
            pass

        # 최종 완료 이벤트 기록
        complexity_summary = {
            "index_type": rag_engine.index_type,
            "num_indexed_chunks": len(rag_engine.documents),
            "expanded_query_count": len(expanded_queries),
            "candidates_evaluated": fusion_result["total_candidates_evaluated"],
            "returned_top_k": len(reranked_docs),
            "time_complexity": {
                "multi_query": "O(M) [M=5]",
                "faiss_dense": faiss_time_str,
                "bm25_sparse": "O(M * |Q_tokens| * N)",
                "rrf_fusion": "O(M * K * log(M * K))",
            },
            "space_complexity": "O(N * d + N * V_vocab + M * K)",
            "latency_breakdown_ms": {
                "multi_query_ms": multi_query_ms,
                "hybrid_search_and_rrf_ms": hybrid_rrf_ms,
                "llm_generation_ms": llm_ms,
                "total_pipeline_ms": total_ms,
            },
        }

        handler.create_event(
            event_type="complete",
            stage="Pipeline Completed",
            data={"message": "RAG Fusion 파이프라인 처리가 완료되었습니다."},
            progress=100,
            complexity={"time": faiss_time_str, "space": "O(N * d)"},
        )

        response_payload = {
            "status": "SUCCESS",
            "original_query": req.query,
            "expanded_queries": expanded_queries,
            "reranked_documents": reranked_docs,
            "answer": answer,
            "sources": [
                {
                    "rank": d["final_rank"],
                    "source": d["source"],
                    "page": d["page"],
                    "chunk_id": d["chunk_id"],
                    "rrf_score": d["rrf_score"],
                }
                for d in reranked_docs
            ],
            "complexity_metrics": complexity_summary,
            "stream_events": handler.events,
        }

        # 클라이언트가 NDJSON 스트리밍을 직접 요청한 경우 StreamingResponse 반환
        if req.stream:
            async def ndjson_generator():
                for ev in handler.events:
                    yield json.dumps(ev, ensure_ascii=False) + "\n"

            return StreamingResponse(
                ndjson_generator(), media_type="application/x-ndjson"
            )

        return response_payload

    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": "ERROR", "error": he.detail},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error": {
                    "error_code": "PIPELINE_EXECUTION_ERROR",
                    "layer": "RAG Fusion Pipeline",
                    "message": str(e),
                    "resolution": "서버 연결 상태 및 입력 파라미터를 점검해 주세요.",
                },
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
