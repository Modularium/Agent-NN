from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib import metadata
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

try:
    dist_path = metadata.distribution("mcp").locate_file("mcp")
    sys.path.insert(0, str(dist_path.parent))
    from mcp.types import BaseModel, Field

    sys.path.pop(0)
except Exception:  # fallback to local pydantic
    from pydantic import BaseModel, Field, field_validator


class AgentRunContext(BaseModel):
    """Result information for a single agent run."""

    agent_id: str
    role: str | None = None
    url: str | None = None
    result: Any | None = None
    subtask_result: Any | None = None
    metrics: Optional[Dict[str, float]] = None
    score: Optional[float] = None
    feedback: Optional[str] = None
    voted_by: List[str] = []


class ModelContext(BaseModel):
    """Context information for a model invocation."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    task: Optional[str] = None
    task_context: "TaskContext | None" = None
    agent_selection: Optional[str] = None
    result: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None
    agents: List["AgentRunContext"] = []
    memory: Optional[List[Dict[str, Any]]] = None
    aggregated_result: Optional[Any] = None
    task_value: Optional[float] = None
    max_tokens: Optional[int] = None
    token_spent: int = 0
    warning: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[int] = None
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    dispatch_state: Literal["queued", "running", "expired", "completed"] = "queued"


class TaskContext(BaseModel):
    """Description and payload for an incoming task."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    task_type: str
    description: Optional[str] = None
    input_data: Any | None = None
    preferences: Optional[Dict[str, Any]] = None

    @field_validator("input_data")
    @classmethod
    def validate_input(cls, v: Any) -> Any:
        limit = int(os.getenv("INPUT_LIMIT_BYTES", "4096"))
        if isinstance(v, dict) and "text" in v and isinstance(v["text"], str):
            if len(v["text"]) > limit:
                raise ValueError("text too long")
        return v


ModelContext.model_rebuild()
AgentRunContext.model_rebuild()
