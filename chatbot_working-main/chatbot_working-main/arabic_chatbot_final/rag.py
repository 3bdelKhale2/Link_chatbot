import os
import asyncio
import time
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from bs4 import BeautifulSoup
import hashlib
import logging

logger = logging.getLogger("rag")


def init_embedding_model(model_name: str):
    model = SentenceTransformer(model_name)
    return model


def get_qdrant_client():
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    COLLECTION_NAME = os.getenv(
        "QDRANT_COLLECTION", "football_ticket_assistant")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
    except Exception as e:
        logger.warning("Could not ensure collection exists: %s", e)
    return client, COLLECTION_NAME


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    bad_patterns = [
        "الصفحة الرئيسية", "البحث في المنتدى", "لوحة تحكم", "الاشتراكات", "المتواجدون الآن",
        "الأقسام", "منتدى", "منتديات", "الشكاوي", "الإقتراحات", "أرشيف"
    ]
    def is_boiler(l: str) -> bool:
        lower = l.lower()
        if sum(1 for bp in bad_patterns if bp in lower) >= 1:
            return True
        if len(l) > 300:
            return True
        return False

    clean = []
    seen = set()
    for l in lines:
        if is_boiler(l):
            continue
        if l in seen:
            continue
        seen.add(l)
        clean.append(l)
    return "\n".join(clean)


def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len > max_chars:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# New: index prepared JSONL chunks into Qdrant
async def index_dataset(jsonl_path: str, model, qdrant_client: QdrantClient, collection_name: str,
                        batch_size: int = 64, limit: int | None = None) -> dict:
    total = 0
    ts = int(time.time())

    def upsert_batch(texts: list[str], metas: list[dict]) -> int:
        if not texts:
            return 0
        vectors = model.encode(texts, show_progress_bar=False)
        points = []
        for txt, v, meta in zip(texts, vectors, metas):
            src = meta.get("source") or ""
            idx = str(meta.get("chunk_index"))
            h = hashlib.sha1((src + "|" + idx + "|" + hashlib.sha1(txt.encode("utf-8")).hexdigest()).encode("utf-8")).hexdigest()
            payload = {"text": txt, **meta, "timestamp": ts}
            points.append({
                # Use stable integer IDs from hash (Qdrant-friendly)
                "id": int(h[:16], 16),
                "vector": v.tolist() if hasattr(v, "tolist") else list(map(float, v)),
                "payload": payload
            })
        qdrant_client.upsert(collection_name=collection_name, points=points)
        return len(points)

    texts, metas = [], []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = (rec.get("text") or "").strip()
                if not txt or len(txt) < 40:
                    continue
                meta = {
                    "source": rec.get("url"),
                    "title": rec.get("title"),
                    "published": rec.get("published"),
                    "chunk_index": rec.get("chunk_index"),
                }
                texts.append(txt)
                metas.append(meta)
                if len(texts) >= batch_size:
                    total += upsert_batch(texts, metas)
                    texts, metas = [], []
                    if limit is not None and total >= limit:
                        break
        # flush remainder
        total += upsert_batch(texts, metas)
        return {"ok": True, "indexed_chunks": total}
    except Exception as e:
        logger.error("Index dataset error: %s", e)
        return {"ok": False, "reason": str(e), "indexed_chunks": total}

def search_rag(model, qdrant_client: QdrantClient, collection_name: str, query: str, top_k: int = 5) -> List[Dict]:
    q_vec = model.encode([query])[0]
    try:
        res = qdrant_client.search(collection_name=collection_name,
                                   query_vector=q_vec.tolist(), limit=top_k, with_payload=True)
    except Exception as e:
        logger.error("Qdrant search error: %s", e)
        return []
    out = []
    for hit in res:
        payload = hit.payload or {}
        text = payload.get("text") or ""
        out.append({"id": hit.id, "score": hit.score,
                   "text": text, "metadata": payload})
    return out
