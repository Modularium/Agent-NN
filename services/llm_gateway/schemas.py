"""Schemas for the LLM Gateway API."""

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    model_name: str | None = None
    temperature: float | None = 0.7


class GenerateResponse(BaseModel):
    completion: str
    tokens_used: int
    provider: str


class EmbedRequest(BaseModel):
    text: str
    model_name: str | None = None


class EmbedResponse(BaseModel):
    embedding: list[float]
    provider: str
