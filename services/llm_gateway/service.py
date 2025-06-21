"""Gateway to various language models."""

from __future__ import annotations

from typing import Any, List

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None


class LLMGatewayService:
    """Generate text via a local LLM pipeline with a fallback."""

    def __init__(self) -> None:
        self.provider = "dummy"
        self.generator = None
        self.embedder = None
        if pipeline is not None:
            try:  # load lightweight default model
                self.generator = pipeline("text-generation", model="distilgpt2")
                self.provider = "local"
            except Exception:  # pragma: no cover - model load can fail
                self.generator = None
        if SentenceTransformer is not None:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self.provider = "local"
            except Exception:  # pragma: no cover - model load can fail
                self.embedder = None

    def generate(
        self, prompt: str, model_name: str | None = None, temperature: float = 0.7
    ) -> dict[str, Any]:
        """Return a text completion using the configured model."""
        if self.generator:
            res = self.generator(
                prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=temperature,
            )[0]
            generated = res["generated_text"][len(prompt) :].strip()
            tokens = len(res["generated_text"].split())
            return {
                "completion": generated,
                "tokens_used": tokens,
                "provider": self.provider,
            }
        # simple fallback if transformers is unavailable
        return {
            "completion": f"Echo: {prompt}",
            "tokens_used": len(prompt.split()),
            "provider": self.provider,
        }

    def embed(self, text: str, model_name: str | None = None) -> dict[str, Any]:
        """Return an embedding for the given text."""
        if self.embedder:
            vec = self.embedder.encode(text)
            emb: List[float] = [float(v) for v in vec]
            return {"embedding": emb, "provider": self.provider}
        return {"embedding": [float(len(text))], "provider": self.provider}

