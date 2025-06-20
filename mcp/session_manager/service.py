from typing import Dict, Optional
import uuid


class SessionManagerService:
    """Simple in-memory session store."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict] = {}

    def create_session(self, data: Dict) -> str:
        sid = str(uuid.uuid4())
        self._sessions[sid] = data
        return sid

    def get_session(self, sid: str) -> Optional[Dict]:
        return self._sessions.get(sid)
