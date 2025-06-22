"""Pydantic models for Task Dispatcher API."""

from core.model_context import TaskContext
from pydantic import Field


class TaskRequest(TaskContext):
    """Incoming task including optional session id."""

    session_id: str | None = Field(default=None)
    mode: str = Field(default="single")
    task_value: float | None = Field(default=None)
    max_tokens: int | None = Field(default=None)
