"""Sample agent worker calling the LLM Gateway."""

from __future__ import annotations

from typing import Any

import httpx

from core.model_context import ModelContext


class SampleAgentService:
    """Process tasks using the LLM Gateway."""

    def __init__(self, llm_url: str = "http://localhost:8003") -> None:
        self.llm_url = llm_url.rstrip("/")

    def run(self, ctx: ModelContext) -> ModelContext:
        """Invoke the LLM Gateway and return the updated context."""
        prompt = ctx.task_context.description or str(ctx.task_context.input_data)
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.llm_url}/generate",
                    json={"prompt": prompt},
                    timeout=10,
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except Exception:
            data = {
                "completion": f"Echo: {prompt}",
                "tokens_used": 0,
                "provider": "dummy",
            }

        ctx.result = data["completion"]
        ctx.metrics = {"tokens_used": data.get("tokens_used", 0)}
        return ctx
