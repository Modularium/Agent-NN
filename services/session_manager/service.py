from __future__ import annotations

"""Session manager backed by a pluggable session store."""

from typing import List

from core.model_context import ModelContext
from core.metrics_utils import ACTIVE_SESSIONS, TASKS_PROCESSED
from core.session_store import BaseSessionStore, InMemorySessionStore


class SessionManagerService:
    """Manage ModelContext sessions via a SessionStore."""

    def __init__(self, store: BaseSessionStore | None = None) -> None:
        self.store = store or InMemorySessionStore()

    def start_session(self) -> str:
        """Create a new session and return its id."""
        sid = self.store.start_session()
        ACTIVE_SESSIONS.labels("session_manager").inc()
        return sid

    def update_context(self, ctx: ModelContext) -> None:
        """Append the given context to its session."""
        if not ctx.session_id:
            ctx.session_id = self.start_session()
        self.store.update_context(ctx.session_id, ctx.model_dump())
        TASKS_PROCESSED.labels("session_manager").inc()

    def get_context(self, session_id: str) -> List[ModelContext]:
        """Return all contexts stored for a session."""
        data = self.store.get_context(session_id)
        TASKS_PROCESSED.labels("session_manager").inc()
        return [ModelContext(**d) for d in data]
