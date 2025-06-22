from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from .model_context import ModelContext


class DispatchQueue:
    """In-memory priority queue for tasks."""

    def __init__(self) -> None:
        self._queue: List[ModelContext] = []

    def _sort(self) -> None:
        def key(ctx: ModelContext):
            deadline = (
                datetime.fromisoformat(ctx.deadline) if ctx.deadline else datetime.max
            )
            priority = ctx.priority if ctx.priority is not None else 5
            value = -(ctx.task_value or 0.0)
            return (deadline, priority, value)

        self._queue.sort(key=key)

    def enqueue(self, ctx: ModelContext) -> str:
        """Add a task context to the queue."""
        if not ctx.submitted_at:
            ctx.submitted_at = datetime.utcnow().isoformat()
        ctx.dispatch_state = "queued"
        self._queue.append(ctx)
        self._sort()
        return ctx.uuid

    def dequeue(self) -> Optional[ModelContext]:
        """Remove and return the next task for execution."""
        if not self._queue:
            return None
        ctx = self._queue.pop(0)
        ctx.dispatch_state = "running"
        return ctx

    def promote_high_priority(self) -> None:
        """Resort the queue by priority."""
        self._sort()

    def expire_old_tasks(self) -> List[ModelContext]:
        """Remove tasks whose deadline is in the past."""
        now = datetime.utcnow()
        remaining: List[ModelContext] = []
        expired: List[ModelContext] = []
        for ctx in self._queue:
            if ctx.deadline and datetime.fromisoformat(ctx.deadline) < now:
                ctx.dispatch_state = "expired"
                expired.append(ctx)
            else:
                remaining.append(ctx)
        self._queue = remaining
        return expired
