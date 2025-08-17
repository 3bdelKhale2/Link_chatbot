# memory.py
from typing import Dict
import time


class SessionMemory:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def _ensure(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "turns": [],
                "slots": {},
                "expectation": None,
                "last_ts": time.time()
            }

    def append_user(self, session_id, text):
        self._ensure(session_id)
        self.sessions[session_id]["turns"].append(
            {"from": "user", "text": text, "ts": time.time()})
        self.sessions[session_id]["last_ts"] = time.time()

    def append_bot(self, session_id, text):
        self._ensure(session_id)
        self.sessions[session_id]["turns"].append(
            {"from": "bot", "text": text, "ts": time.time()})
        self.sessions[session_id]["last_ts"] = time.time()

    def update_slots(self, session_id, slots: dict):
        self._ensure(session_id)
        self.sessions[session_id]["slots"].update(
            {k: v for k, v in slots.items() if v})
        return self.sessions[session_id]["slots"]

    def get_slots(self, session_id):
        self._ensure(session_id)
        return self.sessions[session_id]["slots"]

    def set_expectation(self, session_id, expectation: str):
        self._ensure(session_id)
        self.sessions[session_id]["expectation"] = expectation

    def expectation_is(self, session_id, expectation: str):
        self._ensure(session_id)
        return self.sessions[session_id].get("expectation") == expectation

    def clear_session(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id]["slots"] = {}
            self.sessions[session_id]["expectation"] = None
