"""Simple retrieval agent using the vector store service."""

from __future__ import annotations

import httpx

from core.model_context import ModelContext
from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT


class RetrieverAgent:
    """Retrieve documents related to the task."""

    def __init__(self, vector_url: str = "http://localhost:8004") -> None:
        self.vector_url = vector_url.rstrip("/")

    def run(self, ctx: ModelContext) -> ModelContext:
        query = ctx.task_context.description or ""
        TOKENS_IN.labels("retriever_agent").inc(len(query.split()))
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.vector_url}/vector_search",
                    json={"query": query, "collection": "default", "top_k": 3},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            data = {"matches": []}
        TOKENS_OUT.labels("retriever_agent").inc(len(data.get("matches", [])))
        TASKS_PROCESSED.labels("retriever_agent").inc()
        ctx.result = data.get("matches", [])
        ctx.metrics = {"tokens_used": len(data.get("matches", []))}
        return ctx
