"""Gateway to language model providers."""

from __future__ import annotations

from typing import Any

from core.llm_providers import LLMBackendManager
from core.metrics_utils import TOKENS_IN, TOKENS_OUT
from core.model_context import ModelContext
from services.session_manager.service import SessionManagerService


class LLMGatewayService:
    def __init__(self, manager: LLMBackendManager | None = None) -> None:
        self.manager = manager or LLMBackendManager()
        self.session_mgr = SessionManagerService()

    def chat(self, ctx: ModelContext) -> dict[str, Any]:
        provider_id = self.session_mgr.get_model(ctx.user_id) if ctx.user_id else None
        provider = self.manager.get_provider(provider_id)
        text = provider.generate_response(ctx)
        tokens = len(text.split())
        used = len(ctx.task.split()) if ctx.task else 0
        TOKENS_IN.labels("llm_gateway").inc(used)
        TOKENS_OUT.labels("llm_gateway").inc(tokens)
        return {"completion": text, "provider": provider.name, "tokens_used": tokens}

    def generate(self, prompt: str) -> str:
        ctx = ModelContext(task=prompt)
        return self.chat(ctx)["completion"]

    def embed(self, text: str) -> dict[str, Any]:  # pragma: no cover - optional
        provider = self.manager.get_provider()
        try:
            vector = provider.embed(text)
        except Exception:
            vector = []
        return {"embedding": vector, "provider": provider.name}

    def list_models(self) -> dict[str, Any]:
        return self.manager.available_models()
