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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# RAG utilities and semantic search
from rag import init_embedding_model, get_qdrant_client, index_documents, search_rag
# Intent detection and slot extraction (e.g., day/time for booking)
from intent import detect_intent, extract_booking_slots
# Ticket operations (mock or real API)
from ticketapi import book_ticket, cancel_ticket, list_tickets
# Simple in-memory multi-turn session storage
from memory import SessionMemory
# Structured logger
from logger import init_logger
# LLM wrapper for Arabic text generation
from llm import generate_answer
# External API for today's fixtures
from sports_api import get_today_fixtures

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
app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")

# Configurable models via environment variables
# Embedding model used for vectorization of text chunks and queries
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
# Arabic text generation model used to answer with provided context
TEXT_GEN_MODEL = os.getenv("TEXT_GEN_MODEL", "aubmindlab/aragpt2-base")

# Load embedding model once at startup
MODEL = init_embedding_model(EMBEDDING_MODEL_NAME)

# Connect to Qdrant (vector database) and get collection name
QDRANT_CLIENT, COLLECTION_NAME = get_qdrant_client()

# Per-session memory (stores turns, expected next action, and booking slots)
memory = SessionMemory()


class ChatRequest(BaseModel):
    """Incoming chat message payload."""
    session_id: Optional[str] = None  # Optional session to keep multi-turn context
    text: str                         # User message (Arabic supported)


class CrawlRequest(BaseModel):
    """Request to crawl/index a URL into the vector store for RAG."""
    url: str
    session_id: Optional[str] = None


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

        # ——————————————— Today's fixtures ———————————————
        elif intent == "fixtures.today":
            # Call external API helper and format results
            res = get_today_fixtures()
            if not res.get("ok"):
                if res.get("reason") == "no_token":
                    response = "الرجاء ضبط مفتاح API في المتغير FOOTBALL_DATA_TOKEN لتفعيل نتائج المباريات."
                else:
                    response = "تعذر جلب مباريات اليوم الآن."
            else:
                matches = res.get("matches", [])
                if not matches:
                    response = "لا توجد مباريات اليوم حسب المصدر."
                else:
                    # Limit the list to avoid very long messages
                    lines = [f"- {m['time']}: {m['home']} vs {m['away']}" + (f" ({m['competition']})" if m['competition'] else "")
                             for m in matches[:15]]
                    response = "مباريات اليوم:\n" + "\n".join(lines)

        # ——————————————— RAG-backed fallback ———————————————
        else:
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


@app.post("/crawl")
async def crawl_url(body: CrawlRequest):
    """
    Crawl and index a URL into the RAG vector store.
    Returns number of indexed chunks or a status object.
    """
    result = await index_documents(body.url, MODEL, QDRANT_CLIENT, COLLECTION_NAME)
    return {"indexed": result, "url": body.url}


@app.get("/tickets")
def get_tickets():
    """List all existing tickets (for debugging/admin)."""
    return list_tickets()


@app.get("/")
def root():
    """Simple health/info endpoint."""
    return {"message": "مرحبًا بك في مساعد حجز التذاكر الذكي!"}
