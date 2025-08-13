# intent.py
import re


def _normalize_digits(s: str) -> str:
    # Arabic-Indic => Western digits
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ى", "ي")
    # keep 'ة' as-is for day names like الجمعة; do not convert
    return _normalize_digits(s)


def detect_intent(text: str):
    t = _normalize_text(text)

    # Small talk: greetings/help (rule-based, no LLM)
    if any(k in t for k in ["السلام عليكم", "سلام", "مرحبا", "اهلا", "هاي", "هلا", "hi", "hello"]):
        return "smalltalk.greet", {}
    if any(k in t for k in ["مساعدة", "كيف اساعد", "كيف تساعد", "كيف", "ازاي", "what can you do", "help"]):
        return "smalltalk.help", {}

    # Fixtures today (tolerant of typos/dialect)
    day_words = ["اليوم", "النهارده", "النهاردة", "انهارده"]
    match_words = ["ماتش", "متش", "متشات", "ماتشات", "مباريات", "مباراه", "مباراة", "fixtures"]
    if any(m in t for m in match_words) and any(d in t for d in day_words):
        return "fixtures.today", {}

    # booking
    if any(w in t for w in ["احجز", "حجز", "اريد حجز", "احجزلي", "عايز احجز", "book"]):
        if re.search(r"(الساعة|ساعه|pm|am|\d{1,2})", t) or any(day in t for day in ["الاثنين", "الثلاثاء", "الاربعاء", "الخميس", "الجمعة", "السبت", "الاحد", "اربعاء", "لاربعاء"]):
            return "booking.fill", extract_booking_slots(text)
        return "booking.ask", {}
    # cancel
    if any(w in t for w in ["الغاء", "الغي", "الغى", "الغاء التذكره", "الغاء حجز", "cancel"]):
        m = re.search(r"(?:id[:\s\-]*|معرف[:\s\-]*|#)?([A-Za-z0-9\-]{6,})", text, re.IGNORECASE)
        if m:
            return "cancel.fill", {"ticket_id": m.group(1)}
        return "cancel.ask", {}
    return "general.query", {}


def extract_booking_slots(text: str):
    nt = _normalize_text(text)
    # canonical map
    canon = {
        "الاحد": "الأحد",
        "الاثنين": "الاثنين",
        "الثلاثاء": "الثلاثاء",
        "الاربعاء": "الأربعاء",
        "الخميس": "الخميس",
        "الجمعة": "الجمعة",
        "السبت": "السبت",
    }
    found_day = None
    for k, label in canon.items():
        if k in nt or k.replace("ال", "") in nt or f"ل{k}" in nt:
            found_day = label
            break

    # time: supports Arabic-Indic digits
    m = re.search(r"(\d{1,2}(?::\d{2})?)", nt)
    t = None
    if m:
        t_raw = m.group(1)
        is_pm = any(x in nt for x in ["م", "مساء", "pm"])
        is_am = any(x in nt for x in ["ص", "صباح", "am"])
        if is_pm:
            t = f"{t_raw} مساءً"
        elif is_am:
            t = f"{t_raw} صباحًا"
        else:
            t = t_raw
    return {"day": found_day, "time": t, "raw": text}
