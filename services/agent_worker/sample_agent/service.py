"""Sample agent worker calling the LLM Gateway."""

from __future__ import annotations

from typing import Any

import httpx

from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT

from core.model_context import ModelContext


class SampleAgentService:
    """Process tasks using the LLM Gateway and Vector Store."""

    def __init__(
        self,
        llm_url: str = "http://localhost:8003",
        vector_url: str = "http://localhost:8004",
        session_url: str = "http://localhost:8005",
    ) -> None:
        self.llm_url = llm_url.rstrip("/")
        self.vector_url = vector_url.rstrip("/")
        self.session_url = session_url.rstrip("/")

    def run(self, ctx: ModelContext) -> ModelContext:
        """Invoke the LLM Gateway and return the updated context."""
        prompt = ctx.task_context.description or str(ctx.task_context.input_data)
        task_type = ctx.task_context.task_type if ctx.task_context else ""
        documents: list[dict[str, Any]] = []
        if task_type in {"semantic", "qa", "search"}:
            try:
                with httpx.Client() as client:
                    resp = client.post(
                        f"{self.vector_url}/vector_search",
                        json={"query": prompt, "collection": "default", "top_k": 3},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    documents = resp.json().get("matches", [])
            except Exception:
                documents = []
            doc_text = "\n".join(d.get("text", "") for d in documents)
            prompt = f"{prompt}\n\n{doc_text}" if doc_text else prompt

        TOKENS_IN.labels("sample_agent").inc(len(prompt.split()))

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
        TOKENS_OUT.labels("sample_agent").inc(data.get("tokens_used", 0))
        TASKS_PROCESSED.labels("sample_agent").inc()

        avg_dist = (
            sum(d.get("distance", 0.0) for d in documents) / len(documents)
            if documents
            else 0.0
        )
        ctx.result = {
            "generated_response": data["completion"],
            "sources": documents,
            "embedding_distance_avg": avg_dist,
        }
        ctx.metrics = {"tokens_used": data.get("tokens_used", 0)}
        if ctx.session_id:
            try:
                with httpx.Client() as client:
                    client.post(
                        f"{self.session_url}/update_context",
                        json=ctx.model_dump(),
                        timeout=5,
                    )
            except Exception:
                pass
        return ctx

