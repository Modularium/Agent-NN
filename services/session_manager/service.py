"""Session manager backed by a pluggable session store."""

from __future__ import annotations

from typing import List

from core.model_context import ModelContext
from core.metrics_utils import ACTIVE_SESSIONS, TASKS_PROCESSED
from core.config import settings
from core.session_store import (
    BaseSessionStore,
    InMemorySessionStore,
    FileSessionStore,
)

from core.memory_store import BaseMemoryStore, InMemoryMemoryStore, FileMemoryStore as FileMemoryLogStore, NoOpMemoryStore

class SessionManagerService:
    """Manage ModelContext sessions via a SessionStore."""

    def __init__(
        self,
        store: BaseSessionStore | None = None,
        memory: BaseMemoryStore | None = None,
    ) -> None:
        if store:
            self.store = store
        else:
            if settings.DEFAULT_STORE_BACKEND.lower() == "file":
                self.store = FileSessionStore(settings.SESSIONS_DIR)
            else:
                self.store = InMemorySessionStore()
        if memory:
            self.memory = memory
        else:
            backend = settings.MEMORY_STORE_BACKEND.lower()
            if backend == "file":
                self.memory = FileMemoryLogStore(settings.MEMORY_LOG_DIR)
            elif backend == "noop":
                self.memory = NoOpMemoryStore()
            else:
                self.memory = InMemoryMemoryStore()

    def start_session(self) -> str:
        """Create a new session and return its id."""
        sid = self.store.start_session()
        ACTIVE_SESSIONS.labels("session_manager").inc()
        return sid

    def update_context(self, ctx: ModelContext) -> None:
        """Append the given context to its session and memory log."""
        if not ctx.session_id:
            ctx.session_id = self.start_session()
        entry = {
            "agent": ctx.agent_selection
            or (ctx.agents[-1].agent_id if ctx.agents else None),
            "input": ctx.task_context.description if ctx.task_context else None,
            "output": ctx.result,
            "score": ctx.agents[-1].score if ctx.agents else None,
            "timestamp": ctx.timestamp.isoformat(),
        }
        self.memory.append_memory(ctx.session_id, entry)
        self.store.update_context(ctx.session_id, ctx.model_dump(exclude={"memory"}))
        TASKS_PROCESSED.labels("session_manager").inc()

    def get_context(self, session_id: str) -> List[ModelContext]:
        """Return all contexts stored for a session with aggregated memory."""
        data = self.store.get_context(session_id)
        memory = self.memory.get_memory(session_id)
        TASKS_PROCESSED.labels("session_manager").inc()
        return [ModelContext(**{**d, "memory": memory}) for d in data]
