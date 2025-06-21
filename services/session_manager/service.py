"""Session storage stub."""

from typing import Dict


class SessionManagerService:
    """Manage simple in-memory sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict] = {}

    def get_session(self, session_id: str) -> Dict | None:
        """Return a session by id."""
        return self._sessions.get(session_id)

    def save_session(self, session_id: str, data: Dict) -> None:
        """Persist a session payload."""
        self._sessions[session_id] = data
