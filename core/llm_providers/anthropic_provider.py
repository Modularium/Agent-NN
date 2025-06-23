from __future__ import annotations

from core.model_context import ModelContext

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self.name = "anthropic"
        self.api_key = api_key

    def generate_response(self, ctx: ModelContext) -> str:
        prompt = ctx.task or ""
        return f"anthropic:{prompt}"
