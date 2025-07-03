"""Lightweight orchestrator to manage multiple sessions."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Set, Tuple


class SessionOrchestrator:
    """Schedule and control multiple sessions."""

    def __init__(self) -> None:
        self._queue: Deque[Tuple[int, str]] = deque()
        self._paused: Set[str] = set()

    def schedule_session(self, session_id: str, priority: int = 1) -> None:
        """Add a session to the queue with given priority."""
        self._queue.append((priority, session_id))
        self._queue = deque(sorted(self._queue, reverse=True))

    def pause_session(self, session_id: str) -> None:
        """Pause the given session."""
        self._paused.add(session_id)

    def resume_session(self, session_id: str) -> None:
        """Resume a previously paused session."""
        self._paused.discard(session_id)

    def next_session(self) -> str | None:
        """Return the next runnable session id or ``None``."""
        for _, sid in list(self._queue):
            if sid not in self._paused:
                return sid
        return None

    def status(self) -> Dict[str, str]:
        """Return status info for all sessions."""
        info = {}
        for _, sid in self._queue:
            info[sid] = "paused" if sid in self._paused else "active"
        return info

