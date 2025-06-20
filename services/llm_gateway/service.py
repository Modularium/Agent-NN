"""Gateway to various language models."""

from typing import Any


class LLMGatewayService:
    """Simple echo implementation for text generation."""

    def generate(self, prompt: str) -> dict[str, Any]:
        """Return a dummy text completion."""
        return {"text": f"Echo: {prompt}"}
