from __future__ import annotations

from abc import ABC, abstractmethod

from core.model_context import ModelContext


class LLMProvider(ABC):
    """Abstract interface for language model providers."""

    name: str

    @abstractmethod
    def generate_response(self, ctx: ModelContext) -> str:
        """Generate a completion for the given context."""

    def embed(self, text: str) -> list[float]:  # pragma: no cover - optional
        raise NotImplementedError
