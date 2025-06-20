import time
import uuid
from typing import Dict, Optional


class SessionManagerService:
    """In-memory session store with simple TTL handling."""

    def __init__(self, ttl_minutes: int = 30) -> None:
        self._sessions: Dict[str, Dict] = {}
        self._ttl_seconds = ttl_minutes * 60

    def _cleanup(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s["last_activity"] > self._ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def create_session(self, data: Dict) -> str:
        self._cleanup()
        sid = str(uuid.uuid4())
        self._sessions[sid] = {
            "data": data,
            "history": [],
            "feedback": [],
            "last_activity": time.time(),
        }
        return sid

    def get_session(self, sid: str) -> Optional[Dict]:
        self._cleanup()
        session = self._sessions.get(sid)
        if not session:
            return None
        session["last_activity"] = time.time()
        return session["data"]

    def update_session(self, sid: str, data: Dict) -> None:
        self._cleanup()
        if sid in self._sessions:
            session = self._sessions[sid]
            session["data"].update(data)
            session["last_activity"] = time.time()

    def append_history(self, sid: str, entry: Dict) -> None:
        self._cleanup()
        if sid in self._sessions:
            session = self._sessions[sid]
            session.setdefault("history", []).append(entry)
            session["last_activity"] = time.time()

    def add_feedback(self, sid: str, feedback: Dict) -> None:
        self._cleanup()
        if sid in self._sessions:
            session = self._sessions[sid]
            session.setdefault("feedback", []).append(feedback)
            session["last_activity"] = time.time()

    def get_history(self, sid: str) -> list:
        session = self._sessions.get(sid)
        return session.get("history", []) if session else []

    def get_feedback_stats(self) -> Dict[str, int]:
        total = good = bad = 0
        for session in self._sessions.values():
            for fb in session.get("feedback", []):
                total += 1
                if fb.get("rating") == "good":
                    good += 1
                elif fb.get("rating") == "bad":
                    bad += 1
        return {"total": total, "good": good, "bad": bad}
