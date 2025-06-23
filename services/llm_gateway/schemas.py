"""Schemas for the LLM Gateway API."""

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    completion: str
    tokens_used: int
    provider: str


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    provider: str


class ChatResponse(BaseModel):
    completion: str
    provider: str
    tokens_used: int
