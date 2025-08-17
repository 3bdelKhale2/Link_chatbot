# ticketapi.py
import uuid
import json
import os

DB_FILE = os.getenv("TICKETS_DB", "tickets_db.json")


def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


_db = load_db()


def book_ticket(day: str, time: str):
    ticket_id = str(uuid.uuid4())[:8]
    ticket = {"ticket_id": ticket_id, "day": day, "time": time}
    _db[ticket_id] = ticket
    save_db(_db)
    return ticket


def cancel_ticket(ticket_id: str):
    if ticket_id in _db:
        _db.pop(ticket_id)
        save_db(_db)
        return True
    return False


def list_tickets():
    return list(_db.values())
