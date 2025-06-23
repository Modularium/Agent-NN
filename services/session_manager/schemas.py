"""Pydantic models for the Session Manager API."""

from typing import List

from pydantic import BaseModel

from core.model_context import ModelContext


class SessionId(BaseModel):
    session_id: str


class SessionHistory(BaseModel):
    context: List[ModelContext]


class ModelSelection(BaseModel):
    user_id: str
    model_id: str
