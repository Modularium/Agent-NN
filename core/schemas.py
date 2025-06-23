from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Standard response with status and optional detail."""

    status: str = Field(..., description="Operation status")
    detail: Any | None = Field(default=None, description="Additional information")
