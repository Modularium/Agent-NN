from __future__ import annotations

from core.model_context import ModelContext

from .base import LLMProvider


class LocalHFProvider(LLMProvider):
    def __init__(self, model_path: str) -> None:
        self.name = "local"
        self.model_path = model_path

    def generate_response(self, ctx: ModelContext) -> str:
        prompt = ctx.task or ""
        return f"local:{prompt}"


class GGUFProvider(LLMProvider):
    def __init__(self, model_path: str) -> None:
        self.name = "gguf"
        self.model_path = model_path

    def generate_response(self, ctx: ModelContext) -> str:
        prompt = ctx.task or ""
        return f"gguf:{prompt}"
