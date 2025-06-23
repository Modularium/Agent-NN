"""Session manager backed by a pluggable session store."""

from __future__ import annotations

import os
from typing import List

from core.config import settings
from core.memory_store import (
    BaseMemoryStore,
)
from core.memory_store import FileMemoryStore as FileMemoryLogStore
from core.memory_store import (
    InMemoryMemoryStore,
    NoOpMemoryStore,
)
from core.metrics_utils import (
    ACTIVE_SESSIONS,
    TASKS_PROCESSED,
    FEEDBACK_NEGATIVE,
    FEEDBACK_POSITIVE,
    TASK_SUCCESS,
)
from core.model_context import ModelContext
from core.session_store import BaseSessionStore, FileSessionStore, InMemorySessionStore
from .feedback_store import (
    BaseFeedbackStore,
    FeedbackEntry,
    FileFeedbackStore,
    InMemoryFeedbackStore,
)


class SessionManagerService:
    """Manage ModelContext sessions via a SessionStore."""

    def __init__(
        self,
        store: BaseSessionStore | None = None,
        memory: BaseMemoryStore | None = None,
        feedback: BaseFeedbackStore | None = None,
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
        if feedback:
            self.feedback_store = feedback
        else:
            if settings.DEFAULT_STORE_BACKEND.lower() == "file":
                path = os.path.join(settings.DATA_DIR, "feedback")
                self.feedback_store = FileFeedbackStore(path)
            else:
                self.feedback_store = InMemoryFeedbackStore()
        self._user_models: dict[str, str] = {}

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
            "user_id": ctx.user_id,
            "feedback": ctx.agents[-1].feedback if ctx.agents else None,
            "timestamp": ctx.timestamp.isoformat(),
        }
        self.memory.append_memory(ctx.session_id, entry)
        self.store.update_context(ctx.session_id, ctx.model_dump(exclude={"memory"}))
        TASKS_PROCESSED.labels("session_manager").inc()
        if (
            ctx.task_context
            and ctx.agents
            and ctx.agents[-1].score is not None
            and ctx.agents[-1].score > 0
        ):
            TASK_SUCCESS.labels(ctx.task_context.task_type).inc()

    def get_context(self, session_id: str) -> List[ModelContext]:
        """Return all contexts stored for a session with aggregated memory."""
        data = self.store.get_context(session_id)
        memory = self.memory.get_memory(session_id)
        TASKS_PROCESSED.labels("session_manager").inc()
        return [ModelContext(**{**d, "memory": memory}) for d in data]

    def set_model(self, user_id: str, model_id: str) -> None:
        self._user_models[user_id] = model_id

    def get_model(self, user_id: str | None) -> str | None:
        if not user_id:
            return None
        return self._user_models.get(user_id)

    def add_feedback(self, entry: FeedbackEntry) -> None:
        """Store feedback for a session."""
        self.feedback_store.add_feedback(entry)
        label = entry.agent_id or "unknown"
        if entry.score > 0:
            FEEDBACK_POSITIVE.labels(label).inc()
        else:
            FEEDBACK_NEGATIVE.labels(label).inc()

    def get_feedback(self, session_id: str) -> List[FeedbackEntry]:
        return self.feedback_store.get_feedback(session_id)

    def get_feedback_stats(self) -> dict[str, int]:
        items = self.feedback_store.all_feedback()
        pos = sum(1 for i in items if i.score > 0)
        neg = sum(1 for i in items if i.score <= 0)
        return {"total": len(items), "positive": pos, "negative": neg}
