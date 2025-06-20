"""LLM gateway service wrapping available language models."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List

from llm_models.llm_backend import LLMBackendManager


class LLMGatewayService:
    """Generate text responses and simple QA chains via a unified interface."""

    def __init__(self) -> None:
        self.backend = LLMBackendManager()
        self.vector_url = os.getenv("VECTOR_STORE_URL", "http://vector_store:8000")

    def generate(self, prompt: str) -> str:
        """Return a generated completion for a prompt."""
        llm = self.backend.get_llm()
        try:
            return llm.invoke(prompt)
        except Exception as exc:  # pragma: no cover - network errors
            return f"error: {exc}"

    def qa(self, question: str) -> str:
        """Answer a question using retrieval-augmented generation."""

        docs: List[Dict] = []
        query_url = self.vector_url.rstrip("/") + "/query"
        payload = json.dumps({"query": question}).encode()
        req = urllib.request.Request(
            query_url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                docs = json.loads(resp.read().decode())
        except Exception:
            docs = []

        context = "\n".join(d.get("text", "") for d in docs)
        prompt = (
            "Use the following context to answer the question.\n"
            f"Context:\n{context}\nQuestion: {question}\nAnswer:"
        )
        return self.generate(prompt)
