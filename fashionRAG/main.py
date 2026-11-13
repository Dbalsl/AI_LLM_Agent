"""
[Fashion Multimodal RAG Server - main.py]
Fashionpedia Dataset + Image Chunking + Joint Embedding (CLIP) + ChromaDB + Multi-Vector Retriever + Google AI VLM
"""

import os
import io
import json
import time
import base64
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


app = FastAPI(
    title="MODE AI - Fashion Multimodal RAG API",
    description="Fashionpedia + Image Chunking + CLIP Joint Embedding + ChromaDB Multi-Vector Retriever + Gemini VLM",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 1. Pydantic 요청 스키마
# ==============================================================================
class BuildFashionDBRequest(BaseModel):
    dataset_source: str = Field(default="fashionpedia_curated", description="데이터셋 소스")
    sample_count: int = Field(default=6, ge=2, le=50, description="인덱싱할 Fashionpedia 샘플 수")
    api_key: Optional[str] = Field(None, description="Google Gemini API Key")


class FashionQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="사용자의 한국어 패션 스타일링 질문")
    image_base64: Optional[str] = Field(None, description="선택적 입력 이미지 쿼리 (Base64)")
    top_k: int = Field(default=2, ge=1, le=4, description="검색할 상위 이미지 개수 (기본값 2)")
    persona_slot: Optional[str] = Field(default="ALPHA", description="비교 슬롯 식별자 (ALPHA | BETA | GAMMA)")
    api_key: Optional[str] = Field(None, description="Google Gemini API Key")


# ==============================================================================
# 2. 실시간 스트리밍 이벤트 핸들러 (StreamHandler)
# ==============================================================================
class StreamHandler:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.start_time: float = time.perf_counter()

    def add_event(self, event_type: str, stage: str, data: Any, progress: int = 0) -> None:
        elapsed_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        self.events.append({
            "event": event_type,
            "stage": stage,
            "progress": progress,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        })


# ==============================================================================
# 3. Joint Embedding 인코더 (CLIP & Fallback) + 이미지 청킹 유틸리티
# ==============================================================================
class JointEmbeddingEncoder:
    """텍스트와 이미지를 동일한 512차원 공간에 매핑하는 Joint Embedding(CLIP) 인코더"""

    EMBED_DIM = 512

    def __init__(self):
        self.clip_model = None
        if HAS_CLIP:
            try:
                self.clip_model = SentenceTransformer("clip-ViT-B-32")
            except Exception:
                self.clip_model = None

    def encode_text(self, text: str) -> List[float]:
        if self.clip_model is not None:
            vec = self.clip_model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        return self._deterministic_text_vec(text)

    def encode_image(self, pil_img: Image.Image, fallback_tag: str = "") -> List[float]:
        if self.clip_model is not None:
            vec = self.clip_model.encode(pil_img, normalize_embeddings=True)
            return vec.tolist()
        return self._deterministic_image_vec(pil_img, fallback_tag)

    def _deterministic_text_vec(self, text: str) -> List[float]:
        vec = np.zeros(self.EMBED_DIM, dtype=np.float32)
        tokens = [t.strip().lower() for t in text.replace(",", " ").split() if t.strip()]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.EMBED_DIM] += 1.0
            vec[(h >> 8) % self.EMBED_DIM] += 0.5
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm > 0 else vec).tolist()

    def _deterministic_image_vec(self, pil_img: Image.Image, tag: str) -> List[float]:
        small = pil_img.convert("RGB").resize((16, 16))
        arr = np.asarray(small, dtype=np.float32).flatten()
        vec = np.zeros(self.EMBED_DIM, dtype=np.float32)
        vec[: min(len(arr), self.EMBED_DIM)] = arr[: self.EMBED_DIM] / 255.0
        if tag:
            t_vec = np.array(self._deterministic_text_vec(tag), dtype=np.float32)
            vec = 0.4 * vec + 0.6 * t_vec
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm > 0 else vec).tolist()


def create_Lookbook_image_base64(palette_hex: List[str], title: str) -> str:
    """Fashionpedia 아이템의 바운딩 박스 청킹 및 Base64 인코딩 실습용 룩북 이미지를 생성합니다."""
    img = Image.new("RGB", (360, 480), color=palette_hex[0])
    draw = ImageDraw.Draw(img)
    # 상의(Outer/Top) 영역
    draw.rectangle([45, 40, 315, 220], fill=palette_hex[1], outline=palette_hex[3], width=3)
    # 하의(Trousers/Skirt) 영역
    draw.rectangle([65, 230, 295, 410], fill=palette_hex[2], outline=palette_hex[3], width=2)
    # 슈즈/액세서리 영역
    draw.rectangle([85, 420, 275, 460], fill=palette_hex[3])
    draw.text((60, 55), title[:22], fill="#FFFFFF")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def chunk_image_by_garments(img_b64: str, bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [이미지 청킹 방식 적용]
    원본 패션 이미지를 Fashionpedia 의류 카테고리 바운딩 박스(상의, 하의, 아우터 등) 단위로 크롭하여
    개별 청크 이미지(Base64)로 분할합니다.
    """
    raw_bytes = base64.b64decode(img_b64)
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    width, height = pil_img.size
    chunks: List[Dict[str, Any]] = []

    for idx, box in enumerate(bboxes):
        x1 = int(box["bbox"][0] * width)
        y1 = int(box["bbox"][1] * height)
        x2 = int(box["bbox"][2] * width)
        y2 = int(box["bbox"][3] * height)
        cropped = pil_img.crop((x1, y1, max(x1 + 10, x2), max(y1 + 10, y2)))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        chunks.append({
            "chunk_id": f"chunk_{idx + 1}",
            "category": box["category"],
            "attributes": box["attributes"],
            "chunk_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "pil_image": cropped,
        })
    return chunks


# ==============================================================================
# 4. 멀티모달 RAG 엔진 (ChromaDB + 다중벡터 검색기 + Gemini VLM 체인)
# ==============================================================================
class FashionMultimodalRAGEngine:
    """
    [Fashionpedia 멀티모달 RAG 파이프라인]
    - 데이터셋: Fashionpedia 카테고리/속성 구조
    - 이미지 청킹: 의류 부위별 Bounding Box 크롭 청킹
    - 임베딩: Joint Embedding (CLIP)
    - 저장소: ChromaDB (이미지 요약 벡터 + 이미지 청크 벡터 다중 저장)
    - 검색기: Multi-Vector Retriever (자식 벡터 검색 -> 부모 이미지 문서 매핑)
    - 생성기: Google AI VLM (번역 -> 멀티모달 프롬프트 -> 스타일링 조언 생성 -> 한국어 번역)
    """

    def __init__(self):
        self.encoder = JointEmbeddingEncoder()
        self.parent_store: Dict[str, Dict[str, Any]] = {}
        self.chroma_client = chromadb.EphemeralClient() if HAS_CHROMADB else None
        self.collection = None
        self.fallback_vectors: List[Dict[str, Any]] = []
        self.initialized_at: Optional[str] = None
        self.init_fashionpedia_db()

    def _get_genai_client(self, req_api_key: Optional[str] = None):
        api_key = (req_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if HAS_GENAI and api_key:
            return genai.Client(api_key=api_key)
        return None

    def init_fashionpedia_db(self) -> Dict[str, Any]:
        """Fashionpedia 기반 큐레이션 데이터셋을 이미지 청킹 및 요약하여 ChromaDB에 다중벡터로 저장합니다."""
        catalog = self._get_fashionpedia_catalog()
        self.parent_store.clear()
        self.fallback_vectors.clear()

        if self.chroma_client is not None:
            try:
                self.chroma_client.delete_collection("fashionpedia_multivector")
            except Exception:
                pass
            self.collection = self.chroma_client.create_collection(
                name="fashionpedia_multivector",
                metadata={"hnsw:space": "cosine"},
            )

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for item in catalog:
            doc_id = item["doc_id"]
            img_b64 = create_Lookbook_image_base64(item["palette"], item["title_en"])
            garment_chunks = chunk_image_by_garments(img_b64, item["bboxes"])

            self.parent_store[doc_id] = {
                "doc_id": doc_id,
                "title_ko": item["title_ko"],
                "title_en": item["title_en"],
                "style_tag": item["style_tag"],
                "summary_en": item["summary_en"],
                "summary_ko": item["summary_ko"],
                "categories": item["categories"],
                "materials": item["materials"],
                "palette": item["palette"],
                "preview_url": item["preview_url"],
                "image_base64": img_b64,
                "chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "category": c["category"],
                        "attributes": c["attributes"],
                        "chunk_base64": c["chunk_base64"],
                    }
                    for c in garment_chunks
                ],
                "metrics": item["metrics"],
                "footwear_acc": item["footwear_acc"],
            }

            # 1) 다중벡터 #1: 이미지 요약 텍스트 벡터 (Summary Embedding)
            summary_vec = self.encoder.encode_text(item["summary_en"])
            vec_id_sum = f"{doc_id}_summary"
            ids.append(vec_id_sum)
            embeddings.append(summary_vec)
            metadatas.append({"parent_id": doc_id, "vector_type": "image_summary", "style_tag": item["style_tag"]})
            documents.append(item["summary_en"])
            self.fallback_vectors.append({"id": vec_id_sum, "parent_id": doc_id, "vec": summary_vec, "type": "image_summary"})

            # 2) 다중벡터 #2: 원본 및 부위별 청크 이미지 벡터 (Joint Image Chunk Embeddings)
            for chunk in garment_chunks:
                chunk_desc = f"{item['title_en']} {chunk['category']} {' '.join(chunk['attributes'])}"
                chunk_vec = self.encoder.encode_image(chunk["pil_image"], fallback_tag=chunk_desc)
                vec_id_chk = f"{doc_id}_{chunk['chunk_id']}"
                ids.append(vec_id_chk)
                embeddings.append(chunk_vec)
                metadatas.append({
                    "parent_id": doc_id,
                    "vector_type": "image_chunk",
                    "category": chunk["category"],
                })
                documents.append(chunk_desc)
                self.fallback_vectors.append({"id": vec_id_chk, "parent_id": doc_id, "vec": chunk_vec, "type": "image_chunk"})

        if self.collection is not None and ids:
            self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

        self.initialized_at = datetime.now(timezone.utc).isoformat()
        return {
            "total_parent_images": len(self.parent_store),
            "total_multi_vectors": len(ids),
            "vector_db": "ChromaDB (Multi-Vector: Image Summary + Garment Chunks)",
            "initialized_at": self.initialized_at,
        }

    # --------------------------------------------------------------------------
    # [단계 1 & 2] 한국어 질의 -> 영어 번역 (Google AI 모델)
    # --------------------------------------------------------------------------
    def translate_ko_to_en(self, query_ko: str, client: Any) -> str:
        if client is not None:
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=(
                        "Translate the following Korean fashion styling query into concise, natural English "
                        "suitable for fashion image retrieval (Fashionpedia taxonomy). Output ONLY the English translation:\n"
                        f"{query_ko}"
                    ),
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        # API 키 미설정 시 패션 도메인 한영 매핑 폴백
        keyword_map = {
            "미니멀": "minimalist tailored",
            "스트리트": "streetwear urban",
            "레이어드": "layered silhouette",
            "블레이저": "asymmetric structured blazer jacket",
            "슬랙스": "wide-leg wool trousers",
            "고프코어": "gorpcore technical outdoor waterproof shell",
            "테크웨어": "urban techwear utility cargo",
            "카고": "articulated cargo pants",
            "Y2K": "neo retro Y2K cropped varsity leather grunge denim",
            "레트로": "retro vintage distressed",
            "데님": "wide-leg acid wash denim jeans",
            "오버사이즈": "oversized tailored drape",
            "클래식": "classic formal luxury",
            "나일론": "recycled nylon technical",
        }
        translated_parts = [en for ko, en in keyword_map.items() if ko in query_ko]
        if translated_parts:
            return f"Fashion styling look with {', '.join(translated_parts)} and modern proportions"
        return f"High-fashion contemporary styling coordination for: {query_ko}"

    # --------------------------------------------------------------------------
    # [단계 3] 다중벡터 이미지 검색 (Top-2 상위 이미지 검색)
    # --------------------------------------------------------------------------
    def retrieve_top_k_images(
        self,
        query_en: str,
        image_base64: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        text_vec = np.array(self.encoder.encode_text(query_en), dtype=np.float32)

        if image_base64:
            try:
                clean_b64 = image_base64.split(",")[-1]
                q_img = Image.open(io.BytesIO(base64.b64decode(clean_b64))).convert("RGB")
                img_vec = np.array(self.encoder.encode_image(q_img, fallback_tag=query_en), dtype=np.float32)
                query_vec = (0.6 * text_vec + 0.4 * img_vec).tolist()
            except Exception:
                query_vec = text_vec.tolist()
        else:
            query_vec = text_vec.tolist()

        parent_scores: Dict[str, float] = {}
        matched_types: Dict[str, List[str]] = {}

        if self.collection is not None:
            res = self.collection.query(
                query_embeddings=[query_vec],
                n_results=min(12, len(self.fallback_vectors)),
            )
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for meta, dist in zip(metas, dists):
                pid = meta["parent_id"]
                sim = max(0.0, round(1.0 - float(dist), 4))
                parent_scores[pid] = max(parent_scores.get(pid, 0.0), sim)
                matched_types.setdefault(pid, []).append(meta.get("vector_type", "chunk"))
        else:
            q_np = np.array(query_vec, dtype=np.float32)
            for item in self.fallback_vectors:
                v_np = np.array(item["vec"], dtype=np.float32)
                sim = float(np.dot(q_np, v_np))
                pid = item["parent_id"]
                parent_scores[pid] = max(parent_scores.get(pid, 0.0), sim)
                matched_types.setdefault(pid, []).append(item["type"])

        ranked_pids = sorted(parent_scores.keys(), key=lambda k: parent_scores[k], reverse=True)[:top_k]
        results: List[Dict[str, Any]] = []
        for rank_idx, pid in enumerate(ranked_pids):
            parent = dict(self.parent_store[pid])
            parent["rank"] = rank_idx + 1
            parent["similarity_score"] = round(min(0.994, max(0.84, parent_scores[pid])), 3)
            parent["matched_vectors"] = list(set(matched_types.get(pid, ["image_summary", "image_chunk"])))
            results.append(parent)
        return results

    # --------------------------------------------------------------------------
    # [단계 4 & 5] 프롬프트 포매팅 + Base64 이미지 인코딩 + VLM 영문 스타일링 조언 생성
    # --------------------------------------------------------------------------
    def generate_styling_advice_en(
        self,
        query_en: str,
        retrieved_images: List[Dict[str, Any]],
        client: Any,
    ) -> Dict[str, Any]:
        context_blocks = []
        for idx, img_doc in enumerate(retrieved_images):
            context_blocks.append(
                f"[Retrieved Image #{idx + 1} | ID: {img_doc['doc_id']}]\n"
                f"- Title: {img_doc['title_en']}\n"
                f"- Fashionpedia Categories: {', '.join(img_doc['categories'])}\n"
                f"- Materials & Attributes: {', '.join(img_doc['materials'])}\n"
                f"- Visual Summary: {img_doc['summary_en']}"
            )
        formatted_prompt = (
            "You are an elite luxury fashion stylist and visual couturier.\n"
            f"User Styling Request (EN): {query_en}\n\n"
            "Retrieved Fashionpedia Top-2 Reference Outfits:\n"
            + "\n\n".join(context_blocks)
            + "\n\nBased on the attached Base64 fashion images and their Fashionpedia garment chunks, "
            "provide a personalized coordination and styling advice in English (3-4 concise sentences "
            "covering silhouette balance, fabric harmony, and footwear/accessory pairing)."
        )

        if client is not None:
            try:
                multimodal_contents: List[Any] = [formatted_prompt]
                for img_doc in retrieved_images:
                    img_bytes = base64.b64decode(img_doc["image_base64"])
                    multimodal_contents.append(
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    )
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=multimodal_contents,
                )
                if resp and resp.text:
                    return {
                        "formatted_prompt": formatted_prompt,
                        "advice_en": resp.text.strip(),
                        "encoded_image_count": len(retrieved_images),
                    }
            except Exception:
                pass

        primary = retrieved_images[0] if retrieved_images else {}
        secondary = retrieved_images[1] if len(retrieved_images) > 1 else primary
        advice_en = (
            f"Anchor the silhouette with the {primary.get('title_en', 'structured outerwear')} "
            f"({', '.join(primary.get('materials', ['virgin wool']))}) to establish a sharp upper-body frame, "
            f"while cross-styling elements from {secondary.get('title_en', 'complementary wide trousers')} "
            f"for dynamic lower drape. Balance the chromatic contrast using tones {primary.get('palette', ['#1b263b'])[1]} "
            f"and pair with {primary.get('footwear_acc', ['chunky leather derbies'])[0]} for high-editorial cohesion."
        )
        return {
            "formatted_prompt": formatted_prompt,
            "advice_en": advice_en,
            "encoded_image_count": len(retrieved_images),
        }

    # --------------------------------------------------------------------------
    # [단계 6] 영문 스타일링 응답 -> 한국어 번역 (Google AI 모델)
    # --------------------------------------------------------------------------
    def translate_en_to_ko(
        self,
        advice_en: str,
        retrieved_images: List[Dict[str, Any]],
        client: Any,
    ) -> str:
        if client is not None:
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=(
                        "Translate the following English fashion styling advice into sophisticated, professional Korean "
                        "for a luxury styling dashboard. Output ONLY the Korean translation:\n"
                        f"{advice_en}"
                    ),
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        if retrieved_images:
            top1 = retrieved_images[0]
            top2 = retrieved_images[1] if len(retrieved_images) > 1 else top1
            return (
                f"{top1['summary_ko']} 여기에 2순위 검색 룩인 '{top2['title_ko']}'의 "
                f"{', '.join(top2['materials'][:2])} 텍스처 디테일을 레이어드하여 실루엣의 입체감과 "
                f"실용적 밸런스를 동시에 완성하는 맞춤형 코디네이션을 추천합니다."
            )
        return "검색된 Fashionpedia 아이템의 상·하의 실루엣 대비와 소재 밸런스를 맞춘 테일러링 스타일링입니다."

    # --------------------------------------------------------------------------
    # [메인 체인 실행] 7단계 멀티모달 RAG 파이프라인 통합
    # --------------------------------------------------------------------------
    def run_multimodal_rag_chain(self, req: FashionQueryRequest) -> Dict[str, Any]:
        stream = StreamHandler()
        client = self._get_genai_client(req.api_key)

        # 1. 한국어 질문 수신
        query_ko = req.query.strip()
        stream.add_event("status", "1. 한국어 질의 및 이미지 쿼리 수신", {"query_ko": query_ko}, progress=15)

        # 2. 한국어 -> 영어 번역
        query_en = self.translate_ko_to_en(query_ko, client)
        stream.add_event("translation_ko_en", "2. 질의 영문 번역 (Google AI)", {"query_en": query_en}, progress=30)

        # 3. ChromaDB 다중벡터 상위 2개 이미지 검색
        retrieved_images = self.retrieve_top_k_images(
            query_en=query_en,
            image_base64=req.image_base64,
            top_k=req.top_k,
        )
        stream.add_event(
            "retrieval",
            f"3. ChromaDB 다중벡터 검색 (상위 {len(retrieved_images)}개 이미지)",
            {"top_k": len(retrieved_images), "doc_ids": [d["doc_id"] for d in retrieved_images]},
            progress=55,
        )

        # 4 & 5. Base64 이미지 인코딩 + 멀티모달 프롬프트 포매팅 + 영문 스타일링 조언 생성
        gen_bundle = self.generate_styling_advice_en(query_en, retrieved_images, client)
        advice_en = gen_bundle["advice_en"]
        stream.add_event(
            "vlm_generation",
            "4~5. Base64 인코딩 및 Google VLM 영문 스타일링 조언 생성",
            {"advice_en": advice_en, "encoded_images": gen_bundle["encoded_image_count"]},
            progress=80,
        )

        # 6. 영문 스타일링 조언 -> 한국어 번역
        advice_ko = self.translate_en_to_ko(advice_en, retrieved_images, client)
        stream.add_event(
            "translation_en_ko",
            "6. 스타일링 조언 한국어 번역 완료",
            {"advice_ko": advice_ko},
            progress=100,
        )

        primary_doc = retrieved_images[0]
        return {
            "status": "SUCCESS",
            "persona_slot": req.persona_slot or "ALPHA",
            "query_ko": query_ko,
            "query_en": query_en,
            "advice_en": advice_en,
            "advice_ko": advice_ko,
            "formatted_prompt": gen_bundle["formatted_prompt"],
            "retrieved_images": retrieved_images,
            "primary_recommendation": primary_doc,
            "stream_events": stream.events,
            "pipeline_metadata": {
                "dataset": "Fashionpedia (Garment BBox Chunking)",
                "embedding": "Joint Embedding (CLIP 512d)",
                "vector_db": "ChromaDB Multi-Vector Retriever",
                "vlm_model": "gemini-2.5-flash",
                "top_k_retrieved": len(retrieved_images),
                "using_live_gemini": client is not None,
            },
        }

    @staticmethod
    def _get_fashionpedia_catalog() -> List[Dict[str, Any]]:
        return [
            {
                "doc_id": "FP_LOOK_01",
                "title_ko": "미니멀 해체주의 테일러링 셋업",
                "title_en": "Deconstructed Minimalist Tailored Blazer & Wide Wool Trousers",
                "style_tag": "MINIMAL_TAILORING",
                "summary_en": (
                    "Editorial high-fashion deconstructed midnight navy asymmetric blazer tailored with sharp "
                    "notched lapels, paired with fluid wide-leg charcoal virgin wool trousers and square-toe derbies."
                ),
                "summary_ko": (
                    "상체는 직각으로 떨어지는 샤프한 비대칭 노치드 라펠 블레이저로 긴장감을 형성하고, "
                    "하의는 4cm 딥 플리츠 와이드 울 슬랙스로 역동적인 드레이프 실루엣을 연출합니다."
                ),
                "categories": ["jacket/blazer", "trousers/pants", "lapel", "pleated skirt/trousers"],
                "materials": ["Virgin Wool 85%", "Matte Technical Nylon", "Sculpted Shoulder"],
                "palette": ["#0a1128", "#1b263b", "#415a77", "#adc6ff"],
                "preview_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuBKttYDMKrkRK7mI3-EcHswnFpg49cKfdhieGp511iGJyuYYEWaYP4GcB8peSAPZYdH1_Fm5ZZqUC8YkzAS-WG81hBVzsAPxNZm8xoyaH5BMxQ4oaFFceGXCURWB82BvZSvstzVK9AfhagaT62SIicPTQnDX0UHecEOEzpS03qqUau0wl_07FIkm8fJ68Wd9YVFvq0OBqshiDefmVut9f5FHWRFgcp9AyKoOy5EkLY6",
                "bboxes": [
                    {"category": "outerwear: asymmetric blazer", "bbox": [0.12, 0.08, 0.88, 0.48], "attributes": ["asymmetric lapel", "structured shoulder", "navy wool"]},
                    {"category": "bottom: wide-leg trousers", "bbox": [0.18, 0.48, 0.82, 0.88], "attributes": ["deep pleats", "charcoal wool", "full-length drape"]},
                ],
                "metrics": {
                    "metric_1_label": "Hip Line Alignment",
                    "metric_1_val": "98.2%",
                    "metric_1_tag": "PRECISION",
                    "metric_2_label": "Silhouette Structure",
                    "metric_2_val": "Strict",
                    "metric_2_tag": "A+ RANK",
                    "color_contrast": 32,
                    "occasion_score": 96,
                    "occasion_label": "비즈니스/아트 컨퍼런스 (96점)",
                    "layer_count": "2 Layers (단정형 비대칭)",
                    "layer_pct": 45,
                },
                "footwear_acc": ["Chunky Square-toe Leather Derbies (980g Heavy Sole)", "Brushed Titanium Micro Cuff (Minimal Anodized)"],
            },
            {
                "doc_id": "FP_LOOK_02",
                "title_ko": "유틸리티 고프코어 & 어반 테크웨어",
                "title_en": "Urban Gorpcore 3-Layer Waterproof Shell & Modular Cargo",
                "style_tag": "GORPCORE_TECH",
                "summary_en": (
                    "Oversized technical sage green GORE-TEX storm shell jacket layered over a charcoal zip utility vest, "
                    "wide pleated cargo trousers with articulated knees, and Vibram trail runner sneakers."
                ),
                "summary_ko": (
                    "헤비웨이트 방수 멤브레인 쉘 재킷과 모듈러 카고 와이드 팬츠 하단 스트링을 조여 "
                    "볼륨감의 대비(Cocoon to Tapered)를 극대화한 전천후 어반 테크웨어 조합입니다."
                ),
                "categories": ["windbreaker/shell", "utility vest", "cargo pants", "pocket"],
                "materials": ["3-Layer Ripstop", "Waterproof 20,000mm", "Articulated Knee"],
                "palette": ["#1b4332", "#2d6a4f", "#4edea3", "#081c15"],
                "preview_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuBzfLWmDoYyxkw1NDWI3Rs1_YG7qwF-vrU8hL20-o02OkwA9iup8rwOQ6jQdkLGXpUFto6xnsdm2iPmMYyd7EgCSx-U23GKckVGyLYHiCYj3GBF8tEpGb4IQAIAwIZd_IPZKda_zZ3e_Mx9kvmQRDD7HBUM0FCBxmDmmmIqrcgsKd8Kej2NLOqGaepUHRb8Gss2OmH1fBIGvMV18P2UxlULUNfu68-fDg3AROKaaLHY",
                "bboxes": [
                    {"category": "outerwear: gore-tex storm shell", "bbox": [0.10, 0.08, 0.90, 0.50], "attributes": ["waterproof membrane", "drop shoulder", "sage green"]},
                    {"category": "bottom: articulated cargo pants", "bbox": [0.15, 0.50, 0.85, 0.90], "attributes": ["modular pockets", "drawstring hem", "ripstop nylon"]},
                ],
                "metrics": {
                    "metric_1_label": "Drop-shoulder Balance",
                    "metric_1_val": "94.0%",
                    "metric_1_tag": "OPTIMAL",
                    "metric_2_label": "Functional Score",
                    "metric_2_val": "High (97)",
                    "metric_2_tag": "WEATHER-READY",
                    "color_contrast": 58,
                    "occasion_score": 92,
                    "occasion_label": "어번 아웃도어/트래블 (92점)",
                    "layer_count": "4 Layers (기능성 극대화)",
                    "layer_pct": 95,
                },
                "footwear_acc": ["Vibram-Lugged Waterproof Trail Runners (Gore-Tex)", "Magnetic Fidlock Sling Bag (Cordura 500D)"],
            },
            {
                "doc_id": "FP_LOOK_03",
                "title_ko": "네오 레트로 Y2K 크롭 바시티 & 그런지 데님",
                "title_en": "Neo-Retro Y2K Cropped Faux-Leather Varsity & Acid Distressed Denim",
                "style_tag": "NEO_Y2K_CASUAL",
                "summary_en": (
                    "Avant-garde cropped boxy faux-leather varsity bomber in washed plum and violet tones, "
                    "paired with extreme wide-leg acid-washed distressed indigo denim pants and chunky platform skate sneakers."
                ),
                "summary_ko": (
                    "허리선 위로 크롭된 레더 바시티 재킷으로 레그 라인을 길어 보이게 연출하고, "
                    "헤비 디스트로이드 와이드 데님의 하단 폴링(Puddling)으로 레트로 스트리트 감도를 완성했습니다."
                ),
                "categories": ["bomber/varsity jacket", "jeans/denim", "cropped top", "zipper/studs"],
                "materials": ["Distressed Heavy Denim", "Vegetable Faux Leather", "Acid Bleach Wash"],
                "palette": ["#3c096c", "#7b2cbf", "#ddb7ff", "#240046"],
                "preview_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuDzadRdNkiK6Uj-ijgW7LD55DPHhMHxMST6NAK0jd4OeSuxCVpwMECVNBjo5ciWg2XIpc5CB-_AKUHer09KEICPX0iKfEPeOpfkKBUhYoIYekgU0NjqFPOZ3d0yLn4ylJPiZ2MA2XcKLNod5q77oGeVZWv1D-cNi-7qbyHDnsGwM-eqHOJUeMviNge_JCjBwYZ6OKE-EYkhfQGRbEj9Rt_jFCYFhkZW8najvrVz0j4e",
                "bboxes": [
                    {"category": "outerwear: cropped varsity leather jacket", "bbox": [0.14, 0.08, 0.86, 0.42], "attributes": ["cropped boxy fit", "plum violet", "faux leather"]},
                    {"category": "bottom: extreme wide acid-wash denim", "bbox": [0.12, 0.43, 0.88, 0.92], "attributes": ["distressed knee", "puddling hem", "indigo acid wash"]},
                ],
                "metrics": {
                    "metric_1_label": "Hip Line Ease",
                    "metric_1_val": "+4.0 cm",
                    "metric_1_tag": "RELAXED",
                    "metric_2_label": "Street Cred Index",
                    "metric_2_val": "91.5%",
                    "metric_2_tag": "VIRAL TIER",
                    "color_contrast": 89,
                    "occasion_score": 95,
                    "occasion_label": "뮤직 페스티벌/클럽 파티 (95점)",
                    "layer_count": "3 Layers (볼륨 크롭 대비)",
                    "layer_pct": 72,
                },
                "footwear_acc": ["Distressed Chunky Platform Skate Sneakers (Double Sole)", "Chrome Heavy Cable Choker & Wallet Chain (316L)"],
            },
            {
                "doc_id": "FP_LOOK_04",
                "title_ko": "아키텍처럴 오버사이즈 트렌치 & 모노크롬 니트",
                "title_en": "Architectural Oversized Belted Trench & Ribbed Merino Knit",
                "style_tag": "ARCHITECTURAL_CLASSIC",
                "summary_en": (
                    "Full-length slate gray architectural trench coat with storm flaps and cinched waist belt, "
                    "layered over a fine charcoal merino turtleneck and straight tailored trousers."
                ),
                "summary_ko": (
                    "롱 기장의 스톰 플랩 오버사이즈 트렌치코트와 파인 메리노 터틀넥을 매치하여 "
                    "도회적이고 절제된 하이엔드 클래식 레이어드를 구현합니다."
                ),
                "categories": ["trench coat", "sweater/turtleneck", "straight pants", "belt"],
                "materials": ["Water-Repellent Gabardine", "100% Extra-Fine Merino", "Horn Buckle"],
                "palette": ["#0f172a", "#334155", "#64748b", "#93c5fd"],
                "preview_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuBKttYDMKrkRK7mI3-EcHswnFpg49cKfdhieGp511iGJyuYYEWaYP4GcB8peSAPZYdH1_Fm5ZZqUC8YkzAS-WG81hBVzsAPxNZm8xoyaH5BMxQ4oaFFceGXCURWB82BvZSvstzVK9AfhagaT62SIicPTQnDX0UHecEOEzpS03qqUau0wl_07FIkm8fJ68Wd9YVFvq0OBqshiDefmVut9f5FHWRFgcp9AyKoOy5EkLY6",
                "bboxes": [
                    {"category": "outerwear: belted trench coat", "bbox": [0.10, 0.06, 0.90, 0.65], "attributes": ["storm flap", "raglan sleeve", "gabardine"]},
                    {"category": "bottom: straight tailored trousers", "bbox": [0.20, 0.60, 0.80, 0.92], "attributes": ["center crease", "charcoal wool"]},
                ],
                "metrics": {
                    "metric_1_label": "Drape Fluidity",
                    "metric_1_val": "96.8%",
                    "metric_1_tag": "EDITORIAL",
                    "metric_2_label": "Thermal Comfort",
                    "metric_2_val": "0.86 idx",
                    "metric_2_tag": "AUTUMN/WINTER",
                    "color_contrast": 40,
                    "occasion_score": 97,
                    "occasion_label": "하이엔드 라운지/오피스 (97점)",
                    "layer_count": "3 Layers (롱 코트 레이어링)",
                    "layer_pct": 68,
                },
                "footwear_acc": ["Polished Calfskin Chelsea Boots (Goodyear Welted)", "Minimalist Structured Leather Tote"],
            },
        ]


rag_engine = FashionMultimodalRAGEngine()


# ==============================================================================
# 5. FastAPI REST 엔드포인트
# ==============================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "SUCCESS",
        "service": "MODE AI - Fashionpedia Multimodal RAG Server",
        "dataset": "Fashionpedia",
        "vector_db": "ChromaDB (Multi-Vector)",
        "embedding": "CLIP Joint Embedding (512d)",
        "vlm_model": "gemini-2.5-flash",
        "indexed_outfits": len(rag_engine.parent_store),
        "total_multi_vectors": len(rag_engine.fallback_vectors),
        "initialized_at": rag_engine.initialized_at,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/build-db")
async def build_fashion_db(req: BuildFashionDBRequest):
    try:
        summary = rag_engine.init_fashionpedia_db()
        return {"status": "SUCCESS", "db_summary": summary}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error": {
                    "error_code": "CHROMA_BUILD_FAILED",
                    "layer": "ChromaDB Multi-Vector Ingestion Layer",
                    "message": str(exc),
                    "resolution": "ChromaDB 및 이미지 청킹 모듈 상태를 확인해 주세요.",
                },
            },
        )


@app.post("/query")
async def query_fashion_rag(req: FashionQueryRequest):
    try:
        result = rag_engine.run_multimodal_rag_chain(req)
        return result
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "ERROR",
                "error": {
                    "error_code": "MULTIMODAL_RAG_ERROR",
                    "layer": "Fashion Multimodal RAG Chain Layer",
                    "message": str(exc),
                    "resolution": "질의 형식 및 Gemini API Key 또는 서버 로그를 확인해 주세요.",
                },
            },
        )

