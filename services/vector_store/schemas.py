"""Schemas for the Vector Store API."""

from typing import Any, Dict, List

from pydantic import BaseModel


class AddDocumentRequest(BaseModel):
    text: str
    collection: str


class AddDocumentResponse(BaseModel):
    id: str


class VectorSearchRequest(BaseModel):
    query: str
    collection: str
    top_k: int = 3


class VectorSearchResponse(BaseModel):
    matches: List[Dict[str, Any]]
    model: str
