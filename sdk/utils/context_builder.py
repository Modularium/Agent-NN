"""Helpers to build ModelContext objects."""
from __future__ import annotations

from typing import Optional
from core.model_context import ModelContext, TaskContext


def build_context(task: str, session_id: Optional[str] = None) -> ModelContext:
    """Return a basic ``ModelContext`` for a new task."""
    tc = TaskContext(task_type="generic", description=task)
    return ModelContext(session_id=session_id, task=task, task_context=tc)
