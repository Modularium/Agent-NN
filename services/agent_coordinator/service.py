"""Coordinate multiple agent executions."""

from __future__ import annotations

from typing import List, Tuple

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
                    arc.subtask_result = res.result
                    arc.metrics = res.metrics
                    arc.result = res.result
            ctx.aggregated_result = [a.result for a in ctx.agents]
        elif mode == "voting":
            candidates = [a for a in ctx.agents if a.role != "critic"]
            critics = [a for a in ctx.agents if a.role == "critic"]
            for arc in candidates:
                if arc.url:
                    res = self._call_agent(arc.url, ctx)
                    arc.subtask_result = res.result
                    arc.metrics = res.metrics
                    arc.result = res.result
            for cand in candidates:
                scores: List[float] = []
                feedbacks: List[str] = []
                for critic in critics:
                    if critic.url:
                        score, fb = self._vote_agent(
                            critic.url,
                            str(cand.result),
                            ctx.task_context.description or "",
                            ctx,
                        )
                        if score is not None:
                            scores.append(score)
                        if fb:
                            feedbacks.append(f"{critic.agent_id}:{fb}")
                if scores:
                    cand.score = sum(scores) / len(scores)
                    cand.feedback = " | ".join(feedbacks)
                    cand.voted_by = [c.agent_id for c in critics]
            if candidates:
                best = max(candidates, key=lambda a: a.score or 0.0)
                ctx.aggregated_result = best.result
            else:
                ctx.aggregated_result = None
        else:  # orchestrated pipeline
            data_ctx = ctx
            for role in ["retriever", "summarizer", "writer"]:
                arc = next((a for a in ctx.agents if a.role == role), None)
                if not arc or not arc.url:
                    continue
                res = self._call_agent(arc.url, data_ctx)
                arc.subtask_result = res.result
                arc.metrics = res.metrics
                arc.result = res.result
                data_ctx = res
            ctx.aggregated_result = data_ctx.result
        TASKS_PROCESSED.labels("agent_coordinator").inc()
        tokens = 0
        for a in ctx.agents:
            if a.metrics:
                tokens += int(a.metrics.get("tokens_used", 0))
        TOKENS_OUT.labels("agent_coordinator").inc(tokens)
        ctx.metrics = {"tokens_used": tokens}
        return ctx

    def _call_agent(self, url: str, ctx: ModelContext) -> ModelContext:
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{url.rstrip('/')}/run", json=ctx.model_dump(), timeout=10
                )
                resp.raise_for_status()
                return ModelContext(**resp.json())
        except Exception:
            return ctx

    def _vote_agent(
        self, url: str, text: str, criteria: str, ctx: ModelContext
    ) -> Tuple[float | None, str | None]:
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{url.rstrip('/')}/vote",
                    json={
                        "text": text,
                        "criteria": criteria,
                        "context": ctx.model_dump(),
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("score"), data.get("feedback")
        except Exception:
            return None, None
