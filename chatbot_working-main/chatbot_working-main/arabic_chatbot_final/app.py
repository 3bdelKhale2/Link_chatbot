# app.py
import os
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from rag import init_embedding_model, get_qdrant_client, index_documents, search_rag
from intent import detect_intent, extract_booking_slots
from ticketapi import book_ticket, cancel_ticket, list_tickets
from memory import SessionMemory
from logger import init_logger
from llm import generate_answer
from sports_api import get_today_fixtures

logger = init_logger()
app = FastAPI(title="مساعد حجز تذاكر كرة القدم الذكي")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI at /ui
app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
TEXT_GEN_MODEL = os.getenv("TEXT_GEN_MODEL", "aubmindlab/aragpt2-base")
MODEL = init_embedding_model(EMBEDDING_MODEL_NAME)
QDRANT_CLIENT, COLLECTION_NAME = get_qdrant_client()
memory = SessionMemory()


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    text: str


class CrawlRequest(BaseModel):
    url: str
    session_id: Optional[str] = None


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    user_text = req.text.strip()
    memory.append_user(session_id, user_text)

    intent, details = detect_intent(user_text)
    logger.info({"session_id": session_id, "user": user_text,
                "intent": intent, "details": details})

    response, action = "", None

    try:
        if intent == "booking.ask":
            slots = memory.get_slots(session_id)
            if not slots.get("day") or not slots.get("time"):
                response = "ما هو اليوم والوقت الذي تريد الحجز فيه؟ (مثال: الأربعاء الساعة ٨ مساءً)"
                memory.set_expectation(session_id, "booking")
            else:
                booking = book_ticket(slots["day"], slots["time"])
                response = f"تم الحجز ✅\nمعرف التذكرة: {booking['ticket_id']}"
                memory.clear_session(session_id)
                action = "booked"

        elif intent == "booking.fill":
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

        elif intent == "cancel.ask":
            response = "ما هو معرف التذكرة التي تريد إلغاؤها؟"
            memory.set_expectation(session_id, "cancel")

        elif intent == "cancel.fill":
            ticket_id = details.get(
                "ticket_id") or user_text.strip().split()[-1]
            if cancel_ticket(ticket_id):
                response = f"تم إلغاء التذكرة ✅ (ID: {ticket_id})"
                action = "cancelled"
            else:
                response = f"لم يتم العثور على تذكرة بالمعرف {ticket_id}."

        elif memory.expectation_is(session_id, "booking"):
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
            ticket_id = user_text.strip()
            if cancel_ticket(ticket_id):
                response = f"تم إلغاء التذكرة ✅ (ID: {ticket_id})"
                memory.clear_session(session_id)
                action = "cancelled"
            else:
                response = f"لم يتم العثور على تذكرة بالمعرف {ticket_id}."

        elif intent == "smalltalk.greet":
            response = "وعليكم السلام، كيف أساعدك في الحجز أو معرفة مباريات اليوم؟"

        elif intent == "smalltalk.help":
            response = "أستطيع حجز أو إلغاء تذاكر، وعرض مباريات اليوم. اكتب: حجز، أو: ما مباريات اليوم."

        elif intent == "fixtures.today":
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
                    lines = [f"- {m['time']}: {m['home']} vs {m['away']}" + (f" ({m['competition']})" if m['competition'] else "")
                             for m in matches[:15]]
                    response = "مباريات اليوم:\n" + "\n".join(lines)

        else:
            docs = search_rag(MODEL, QDRANT_CLIENT,
                              COLLECTION_NAME, user_text, top_k=5)
            if not docs:
                # No trusted context → don't use the LM to avoid hallucinations
                response = "عذرًا، لم أجد معلومات متعلقة بسؤالك."
            else:
                # Keep context small and relevant
                context = "\n\n".join([d['text'] for d in docs])[:1500]
                try:
                    response = generate_answer(TEXT_GEN_MODEL, user_text, context)
                except Exception as e:
                    logger.error(f"LLM generation error: {e}")
                    response = "عذرًا، لم أجد معلومات متعلقة بسؤالك."

    except Exception as e:
        logger.error(f"Error during chat handling: {e}")
        response = "حدث خطأ أثناء معالجة طلبك. يرجى المحاولة لاحقًا."

    memory.append_bot(session_id, response)
    logger.info({"session_id": session_id, "bot": response, "action": action})
    return {"session_id": session_id, "response": response, "intent": intent, "action": action}


@app.post("/crawl")
async def crawl_url(body: CrawlRequest):
    result = await index_documents(body.url, MODEL, QDRANT_CLIENT, COLLECTION_NAME)
    return {"indexed": result, "url": body.url}


@app.get("/tickets")
def get_tickets():
    return list_tickets()


@app.get("/")
def root():
    return {"message": "مرحبًا بك في مساعد حجز التذاكر الذكي!"}
