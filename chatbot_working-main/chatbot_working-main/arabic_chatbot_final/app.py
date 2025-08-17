# app.py
"""
Arabic football ticket assistant API.

Capabilities:
- Book and cancel tickets via simple intents and slot filling (day/time).
- Answer small talk/help.
- Show today's fixtures via external sports API.
- RAG fallback: search indexed content (Qdrant) and generate grounded answers.
- Serve a static web UI from ./web at /ui.
"""

import os
import uuid
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict

# RAG utilities and semantic search
from .rag import init_embedding_model, get_qdrant_client, search_rag, index_dataset
# Intent detection and slot extraction (e.g., day/time for booking)
from .intent import detect_intent, extract_booking_slots
# Ticket operations (mock or real API)
from .ticketapi import book_ticket, cancel_ticket, list_tickets
# Simple in-memory multi-turn session storage
from .memory import SessionMemory
# Structured logger
from .logger import init_logger
# LLM wrapper for Arabic text generation
from .llm import generate_answer

# Initialize application-wide logger
logger = init_logger()

# Create FastAPI app with Arabic title
app = FastAPI(title="مساعد حجز تذاكر كرة القدم الذكي")

# Allow cross-origin requests (frontend access). Note:
# Using allow_origins=["*"] with allow_credentials=True is blocked by browsers.
# If you need credentials, specify explicit origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI (static files) under /ui from ./web directory
WEB_DIR = Path(__file__).resolve().parent / "web"
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")
else:
    logger.warning(f"Static UI directory not found at {WEB_DIR}. Skipping /ui mount.")

# Configurable models via environment variables
# Embedding model used for vectorization of text chunks and queries
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
# Arabic text generation model used to answer with provided context
TEXT_GEN_MODEL = os.getenv("TEXT_GEN_MODEL", "aubmindlab/aragpt2-base")

BASE_DIR = Path(__file__).resolve().parent
# Prefer env var; otherwise default to ./data under this module
DEFAULT_CHUNKS_PATH = os.getenv("CHUNKS_JSONL") or str(BASE_DIR / "data" / "yallakora_articles_chunked.jsonl")

# Load embedding model once at startup
MODEL = init_embedding_model(EMBEDDING_MODEL_NAME)

# Connect to Qdrant (vector database) and get collection name
QDRANT_CLIENT, COLLECTION_NAME = get_qdrant_client()

# Per-session memory (stores turns, expected next action, and booking slots)
memory = SessionMemory()

# =========================
# Matches dataset (JSONL)
# =========================

DATA_PATH = Path(r"c:\Users\user\Desktop\LinkDev_Intern\chatbot_working-main\chatbot_working-main\data\matches_clean.jsonl")

def _clean_text(s: str) -> str:
    if not isinstance(s, str): return s
    s = re.sub(r'\s*\|\s*يلاكورة\s*$', '', s)   # إزالة | يلاكورة
    s = s.replace('ـ', '')                       # إزالة التطويل
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_matches_by_date(path=DATA_PATH):
    by_date = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):  # تخطي التعليقات
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # تنظيف الحقول المهمة
            obj['home'] = _clean_text(obj.get('home', ''))
            obj['away'] = _clean_text(obj.get('away', ''))
            obj['competition'] = _clean_text(obj.get('competition', ''))
            by_date[obj.get('date', '')].append(obj)
    return dict(by_date)

MATCHES_BY_DATE = load_matches_by_date()

def _parse_ar_date(user_text: str, now=None):
    now = now or datetime.now()
    t = user_text or ''
    t_norm = re.sub(r'\s+', ' ', t).strip()
    # كلمات شائعة
    if 'اليوم' in t_norm:
        return now.date()
    if any(w in t_norm for w in ['بكره','بكرة','غدا','غداً','غدوة']):
        return (now + timedelta(days=1)).date()
    if any(w in t_norm for w in ['أمس','امس','بارح']):
        return (now - timedelta(days=1)).date()
    # صيغ تواريخ شائعة: 8/14/2025 أو 14/8/2025 أو 2025-08-14
    m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', t_norm)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d).date()
    m = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', t_norm)
    if m:
        a, b, c = m.groups()
        a, b, c = int(a), int(b), int(c)
        # نحاول تخمين هل التنسيق MM/DD أم DD/MM: لو a>12 نفترض DD/MM
        if c < 100: c += 2000
        if a > 12:  # DD/MM/YYYY
            d, mo, y = a, b, c
        else:       # نفترض MM/DD/YYYY
            mo, d, y = a, b, c
        return datetime(y, mo, d).date()
    return None

def _format_matches(matches, include_urls: bool = False):
    # إخراج مختصر بدون روابط
    lines = []
    for m in sorted(matches, key=lambda x: x.get('time','')):
        time = m.get('time', '')
        home = m.get('home', '')
        away = m.get('away', '')
        comp = m.get('competition','')
        if include_urls and m.get('url'):
            url = m.get('url','')
            lines.append(f"- {time} | {home} ضد {away} — {comp} — {url}")
        else:
            lines.append(f"- {time} | {home} ضد {away} — {comp}")
    return "\n".join(lines)

def find_matches_message(user_text: str, now=None) -> str:
    target_date = _parse_ar_date(user_text, now=now)
    if not target_date:
        return "من فضلك حدّد التاريخ (مثال: اليوم، بكرة، أو 2025-12-26)."

    key = target_date.isoformat()
    matches = MATCHES_BY_DATE.get(key, [])

    if matches:
        header = f"مباريات يوم {key} ({len(matches)} مباراة):"
        return header + "\n" + _format_matches(matches)

    # لا توجد مباريات لذلك اليوم: نعرض أقرب يوم متاح بعده
    future_dates = sorted(d for d in MATCHES_BY_DATE.keys() if d > key)
    past_dates   = sorted((d for d in MATCHES_BY_DATE.keys() if d < key), reverse=True)

    if future_dates:
        next_day = future_dates[0]
        nxt = MATCHES_BY_DATE[next_day]
        msg = f"لا توجد مباريات بتاريخ {key} في قاعدة البيانات.\nأقرب يوم فيه مباريات: {next_day} ({len(nxt)} مباراة):"
        return msg + "\n" + _format_matches(nxt[:10])  # نعرض أول 10 إن كانت كثيرة
    if past_dates:
        prev_day = past_dates[0]
        prv = MATCHES_BY_DATE[prev_day]
        msg = f"لا توجد مباريات بتاريخ {key} في قاعدة البيانات.\nآخر يوم مضى وفيه مباريات: {prev_day} ({len(prv)} مباراة):"
        return msg + "\n" + _format_matches(prv[:10])

    return "قاعدة البيانات فارغة أو لم تُحمّل بشكل صحيح."

class ChatRequest(BaseModel):
    """Incoming chat message payload."""
    session_id: Optional[str] = None  # Optional session to keep multi-turn context
    text: str                         # User message (Arabic supported)


class IndexRequest(BaseModel):
    """Request to bulk index a prepared JSONL file into the vector store."""
    path: str = "data/yallakora_articles_chunked.jsonl"
    limit: Optional[int] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main chat handler.

    - Creates or reuses a session_id.
    - Detects intent and routes to booking/cancel/help/fixtures/RAG fallback.
    - Persists conversation to memory.
    - Returns:
      { session_id, response, intent, action? }
      where action is "booked" or "cancelled" on success.
    """
    # Ensure we have a session ID for multi-turn flows
    session_id = req.session_id or str(uuid.uuid4())
    user_text = req.text.strip()

    # Store user turn in memory
    memory.append_user(session_id, user_text)

    # Detect high-level intent and extract any details (entities)
    intent, details = detect_intent(user_text)
    logger.info({"session_id": session_id, "user": user_text,
                "intent": intent, "details": details})

    response, action = "", None

    try:
        # ——————————————— Booking flows ———————————————
        if intent == "booking.ask":
            # User asked to book. Check if we already have required slots.
            slots = memory.get_slots(session_id)
            if not slots.get("day") or not slots.get("time"):
                # Ask user to provide missing day/time and set expectation
                response = "ما هو اليوم والوقت الذي تريد الحجز فيه؟ (مثال: الأربعاء الساعة ٨ مساءً)"
                memory.set_expectation(session_id, "booking")
            else:
                # If slots already present, proceed to book immediately
                booking = book_ticket(slots["day"], slots["time"])
                response = f"تم الحجز ✅\nمعرف التذكرة: {booking['ticket_id']}"
                memory.clear_session(session_id)
                action = "booked"

        elif intent == "booking.fill":
            # Extract slots (day/time) from the user text and update memory
            slots = extract_booking_slots(user_text)
            memory.update_slots(session_id, slots)
            slots_now = memory.get_slots(session_id)

            # If we have both slots, complete the booking
            if slots_now.get("day") and slots_now.get("time"):
                booking = book_ticket(slots_now["day"], slots_now["time"])
                response = f"تم الحجز ✅\nمعرف التذكرة: {booking['ticket_id']}"
                memory.clear_session(session_id)
                action = "booked"
            else:
                # Ask the user to clarify
                response = "يرجى تحديد اليوم والوقت بشكل أوضح."

        # ——————————————— Cancel flows ———————————————
        elif intent == "cancel.ask":
            # Ask for ticket ID and set expectation to cancel
            response = "ما هو معرف التذكرة التي تريد إلغاؤها؟"
            memory.set_expectation(session_id, "cancel")

        elif intent == "cancel.fill":
            # Try to get the ticket ID either from extracted details or from the message
            ticket_id = details.get("ticket_id") or user_text.strip().split()[-1]
            if cancel_ticket(ticket_id):
                response = f"تم إلغاء التذكرة ✅ (ID: {ticket_id})"
                action = "cancelled"
            else:
                response = f"لم يتم العثور على تذكرة بالمعرف {ticket_id}."

        # ——————————————— Expectation-based continuations ———————————————
        elif memory.expectation_is(session_id, "booking"):
            # Continue slot filling for booking
            slots = extract_booking_slots(user_text)
            memory.update_slots(session_id, slots)
            slots_now = memory.get_slots(session_id)
            if slots_now.get("day") and slots_now.get("time"):
                booking = book_ticket(slots_now["day"], slots_now["time"])
                response = f"تم الحجز ✅\nمعرف التذكرة: {booking['ticket_id']}"
                memory.clear_session(session_id)
                action = "booked"
            else:
                response = "يرجى تحديد اليوم والوقت بشكل أوضح."

        elif memory.expectation_is(session_id, "cancel"):
            # Treat current message as ticket ID and attempt cancellation
            ticket_id = user_text.strip()
            if cancel_ticket(ticket_id):
                response = f"تم إلغاء التذكرة ✅ (ID: {ticket_id})"
                memory.clear_session(session_id)
                action = "cancelled"
            else:
                response = f"لم يتم العثور على تذكرة بالمعرف {ticket_id}."

        # ——————————————— Small talk/help ———————————————
        elif intent == "smalltalk.greet":
            response = "وعليكم السلام، كيف أساعدك في الحجز أو معرفة مباريات اليوم؟"

        elif intent == "smalltalk.help":
            response = "أستطيع حجز أو إلغاء تذاكر، وعرض مباريات اليوم. اكتب: حجز، أو: ما مباريات اليوم."

        # ——————————————— Matches dataset query ———————————————
        else:
            # Try to answer using matches_clean.jsonl
            matches_ans = find_matches_message(user_text)
            if matches_ans:
                response = matches_ans
            else:
                # ——————————————— RAG-backed fallback ———————————————
                # Retrieve relevant chunks from vector DB
                docs = search_rag(MODEL, QDRANT_CLIENT, COLLECTION_NAME, user_text, top_k=5)
                if not docs:
                    # No trusted context → avoid LLM to reduce hallucinations
                    response = "عذرًا، لم أجد معلومات متعلقة بسؤالك."
                else:
                    # Keep context small and relevant to control generation
                    context = "\n\n".join([d['text'] for d in docs])[:1500]
                    try:
                        response = generate_answer(TEXT_GEN_MODEL, user_text, context)
                    except Exception as e:
                        logger.error(f"LLM generation error: {e}")
                        response = "عذرًا، لم أجد معلومات متعلقة بسؤالك."

    except Exception as e:
        # Catch-all to prevent 500s from leaking
        logger.error(f"Error during chat handling: {e}")
        response = "حدث خطأ أثناء معالجة طلبك. يرجى المحاولة لاحقًا."

    # Store bot turn and log final outcome
    memory.append_bot(session_id, response)
    logger.info({"session_id": session_id, "bot": response, "action": action})

    # Return structured response for the frontend
    return {"session_id": session_id, "response": response, "intent": intent, "action": action}


class MatchesSearchRequest(BaseModel):
    q: str
    limit: Optional[int] = 8

@app.post("/matches/search")
def matches_search(body: MatchesSearchRequest):
    """
    Search matches dataset using free-text Arabic query.
    Examples:
      - "مباريات اليوم"
      - "مباراة ليفربول 2024-12-14"
      - "مباريات الدوري الفرنسي 2024-09-28"
    """
    ans = find_matches_message(body.q)
    return {"results": ans or "لا توجد نتائج"}


@app.post("/index")
async def index_dataset_endpoint(body: IndexRequest):
    """
    Bulk index prepared chunks (JSONL) into the vector store.
    """
    # Normalize provided path: relative to this file if not absolute
    p = Path(body.path)
    if not p.is_absolute():
        # Try ./<path> then ../<path>
        cand1 = BASE_DIR / p
        cand2 = BASE_DIR.parent / p
        p = cand1 if cand1.exists() else cand2
    result = await index_dataset(str(p), MODEL, QDRANT_CLIENT, COLLECTION_NAME, limit=body.limit)
    return {"indexed": result, "path": str(p)}


@app.get("/tickets")
def get_tickets():
    """List all existing tickets (for debugging/admin)."""
    return list_tickets()


@app.get("/")
def root():
    """Simple health/info endpoint."""
    return {"message": "مرحبًا بك في مساعد حجز التذاكر الذكي!"}


@app.on_event("startup")
async def ensure_indexed_startup():
    # Log matches dataset status on startup
    if MATCHES_BY_DATE:
        logger.info(f"Loaded matches dataset with {len(MATCHES_BY_DATE)} dates.")
    else:
        logger.warning("Matches dataset not found or empty.")
    try:
        cnt = QDRANT_CLIENT.count(COLLECTION_NAME, exact=True).count
        if not cnt or cnt == 0:
            # Resolve default path with fallbacks
            p = Path(DEFAULT_CHUNKS_PATH)
            if not p.exists():
                alt = BASE_DIR.parent / "data" / "yallakora_articles_chunked.jsonl"
                if alt.exists():
                    p = alt
            if not p.exists():
                logger.warning(f"Chunks file not found at {DEFAULT_CHUNKS_PATH}. "
                               f"Tried also {BASE_DIR.parent / 'data' / 'yallakora_articles_chunked.jsonl'}. Skipping auto-index.")
                return
            logger.info(f"Index empty. Indexing chunks from {p} ...")
            res = await index_dataset(str(p), MODEL, QDRANT_CLIENT, COLLECTION_NAME)
            logger.info({"index_result": res})
        else:
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already has {cnt} points.")
    except Exception as e:
        logger.error(f"Startup indexing failed: {e}")
