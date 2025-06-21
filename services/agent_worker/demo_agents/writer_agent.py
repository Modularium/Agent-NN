"""Simple writer agent using the LLM gateway."""

from __future__ import annotations

import httpx

from core.model_context import ModelContext
from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT


class WriterAgent:
    """Generate a text completion from a prompt."""

    def __init__(self, llm_url: str = "http://localhost:8003") -> None:
        self.llm_url = llm_url.rstrip("/")

    def run(self, ctx: ModelContext) -> ModelContext:
        prompt = ctx.task_context.description or ""
        if ctx.memory:
            recent = [m.get("output", "") for m in ctx.memory[-2:] if isinstance(m.get("output"), str)]
            if recent:
                prompt = " ".join(recent) + "\n" + prompt
        if ctx.task_context.input_data:
            prompt = f"{prompt}\n{ctx.task_context.input_data}"
        if isinstance(ctx.result, list):
            docs = "\n".join(d.get("text", "") for d in ctx.result)
            prompt = f"{prompt}\n\n{docs}" if docs else prompt
        TOKENS_IN.labels("writer_agent").inc(len(prompt.split()))
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.llm_url}/generate",
                    json={"prompt": prompt},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            data = {"completion": prompt, "tokens_used": 0}
        TOKENS_OUT.labels("writer_agent").inc(data.get("tokens_used", 0))
        TASKS_PROCESSED.labels("writer_agent").inc()
        ctx.result = data.get("completion")
        ctx.metrics = {"tokens_used": data.get("tokens_used", 0)}
        return ctx
