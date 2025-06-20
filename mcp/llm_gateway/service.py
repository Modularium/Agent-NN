"""LLM gateway service wrapping available language models."""

from __future__ import annotations

import json
import os
import urllib.request
import logging
from typing import Dict, List

from llm_models.llm_backend import LLMBackendManager


class LLMGatewayService:
    """Generate text responses and simple QA chains via a unified interface."""

    def __init__(self) -> None:
        self.backend = LLMBackendManager()
        self.vector_url = os.getenv("VECTOR_STORE_URL", "http://vector_store:8000")
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def generate(self, prompt: str) -> str:
        """Return a generated completion for a prompt."""
        llm = self.backend.get_llm()
        try:
            self.logger.info("LLM generate: %s", prompt)
            return llm.invoke(prompt)
        except Exception as exc:  # pragma: no cover - network errors
            self.logger.error("LLM generate failed: %s", exc)
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
            self.logger.info("Query vector store: %s", question)
            with urllib.request.urlopen(req, timeout=10) as resp:
                docs = json.loads(resp.read().decode())
        except Exception:
            self.logger.error("Vector store query failed", exc_info=True)
            docs = []

        context = "\n".join(d.get("text", "") for d in docs)
        prompt = (
            "Use the following context to answer the question.\n"
            f"Context:\n{context}\nQuestion: {question}\nAnswer:"
        )
        return self.generate(prompt)

    def translate(self, text: str, target_lang: str) -> str:
        """Return a translated version of text using the LLM backend."""
        prompt = f"Translate the following text to {target_lang}:\n{text}"
        return self.generate(prompt)

    def vision_describe(self, image_url: str) -> str:
        """Describe an image. Currently a placeholder for future multimodal models."""
        prompt = f"Describe the image at {image_url}."
        return self.generate(prompt)
