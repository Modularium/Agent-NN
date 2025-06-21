"""Schemas for the Vector Store API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class VectorSearchRequest(BaseModel):
    text: str
    embedding_model: Optional[str] = None


class VectorSearchResponse(BaseModel):
    matches: List[Dict[str, Any]]
    model: str
