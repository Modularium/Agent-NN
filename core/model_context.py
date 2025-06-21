from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

"""MCP-compatible context models."""

from importlib import metadata, util
import sys

try:
    dist_path = metadata.distribution("mcp").locate_file("mcp")
    sys.path.insert(0, str(dist_path.parent))
    from mcp.types import BaseModel, Field

    sys.path.pop(0)
except Exception:  # fallback to local pydantic
    from pydantic import BaseModel, Field


class ModelContext(BaseModel):
    """Context information for a model invocation."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session: Optional[str] = None
    task: Optional[str] = None
    task_context: "TaskContext | None" = None
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


ModelContext.model_rebuild()
