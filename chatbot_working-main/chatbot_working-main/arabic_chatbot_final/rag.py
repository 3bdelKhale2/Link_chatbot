# rag.py
import os
import asyncio
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from crawl4ai import AsyncWebCrawler
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


async def crawl_page(url: str):
    crawler = AsyncWebCrawler(api_key=os.getenv("CRAWL4AI_KEY", ""))
    try:
        result = await crawler.crawl(urls=[url], max_pages=1)
        html = result[0].get("html", "") if result else ""
        return html
    except Exception as e:
        logger.error("Crawl error: %s", e)
        return ""


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Drop common navigation/boilerplate lines
    bad_patterns = [
        "الصفحة الرئيسية", "البحث في المنتدى", "لوحة تحكم", "الاشتراكات", "المتواجدون الآن",
        "الأقسام", "منتدى", "منتديات", "الشكاوي", "الإقتراحات", "أرشيف"
    ]
    def is_boiler(l: str) -> bool:
        lower = l.lower()
        if sum(1 for bp in bad_patterns if bp in lower) >= 1:
            return True
        if len(l) > 300:  # overly long menus
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


async def index_documents(url: str, model, qdrant_client: QdrantClient, collection_name: str):
    html = await crawl_page(url)
    if not html:
        return {"ok": False, "reason": "crawl_failed"}
    text = html_to_text(html)
    chunks = chunk_text(text, max_chars=800)
    vectors = model.encode(chunks, show_progress_bar=False)
    points = []
    ts = int(time.time())
    for i, (chunk, v) in enumerate(zip(chunks, vectors)):
        uid = hashlib.sha1((url + str(i)).encode()).hexdigest()
        metadata = {"source": url, "chunk_index": i, "timestamp": ts}
        points.append({
            "id": uid,
            "vector": v.tolist() if hasattr(v, "tolist") else list(map(float, v)),
            "payload": {"text": chunk, **metadata}
        })
    try:
        qdrant_client.upsert(collection_name=collection_name, points=points)
        return {"ok": True, "indexed_chunks": len(points)}
    except Exception as e:
        logger.error("Qdrant upsert error: %s", e)
        return {"ok": False, "reason": str(e)}


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
