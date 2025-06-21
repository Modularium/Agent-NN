from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ModelContext(BaseModel):
    """Context information for a model invocation."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session: Optional[str] = None
    task: Optional[str] = None
    agent_selection: Optional[str] = None
    result: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None


class TaskContext(BaseModel):
    """Description and payload for an incoming task."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: str
    description: Optional[str] = None
    input_data: Any | None = None
    preferences: Optional[Dict[str, Any]] = None
