"""Coordinate multiple agent executions."""

from __future__ import annotations

from typing import List

import httpx

from core.model_context import ModelContext, AgentRunContext
from core.metrics_utils import TASKS_PROCESSED, TOKENS_OUT


class AgentCoordinatorService:
    """Run agents in parallel or orchestrated mode."""

    def __init__(self) -> None:
        pass

    def coordinate(self, ctx: ModelContext, mode: str = "parallel") -> ModelContext:
        if mode == "parallel":
            for arc in ctx.agents:
                if arc.url:
                    res = self._call_agent(arc.url, ctx)
                    arc.result = res.result
                    arc.metrics = res.metrics
            ctx.aggregated_result = [a.result for a in ctx.agents]
        else:  # orchestrated pipeline
            data_ctx = ctx
            for role in ["retriever", "summarizer", "writer"]:
                arc = next((a for a in ctx.agents if a.role == role), None)
                if not arc or not arc.url:
                    continue
                res = self._call_agent(arc.url, data_ctx)
                arc.result = res.result
                arc.metrics = res.metrics
                data_ctx = res
            ctx.aggregated_result = data_ctx.result
        TASKS_PROCESSED.labels("agent_coordinator").inc()
        tokens = 0
        for a in ctx.agents:
            if a.metrics:
                tokens += int(a.metrics.get("tokens_used", 0))
        TOKENS_OUT.labels("agent_coordinator").inc(tokens)
        return ctx

    def _call_agent(self, url: str, ctx: ModelContext) -> ModelContext:
        try:
            with httpx.Client() as client:
                resp = client.post(f"{url.rstrip('/')}/run", json=ctx.model_dump(), timeout=10)
                resp.raise_for_status()
                return ModelContext(**resp.json())
        except Exception:
            return ctx
