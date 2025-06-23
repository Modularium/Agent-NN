from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Standard response with status and optional detail."""

    status: str = Field(..., description="Operation status")
    detail: Any | None = Field(default=None, description="Additional information")


class ErrorResponse(BaseModel):
    """Error details with message and optional code."""

    status: str = Field(default="error", description="Error status")
    detail: str = Field(..., description="Error message")
    code: int | None = Field(default=None, description="Optional error code")
