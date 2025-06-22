"""Task dispatcher core logic."""

import random
import time
from typing import Any, List

import httpx

from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT

from core.model_context import ModelContext, TaskContext, AgentRunContext
from .config import settings


class TaskDispatcherService:
    """Dispatch incoming tasks to worker agents."""

    def __init__(
        self,
        registry_url: str | None = None,
        session_url: str | None = None,
        coordinator_url: str | None = None,
        coalition_url: str | None = None,
    ) -> None:
        self.registry_url = (registry_url or settings.registry_url).rstrip("/")
        self.session_url = (session_url or settings.session_url).rstrip("/")
        self.coordinator_url = (coordinator_url or settings.coordinator_url).rstrip("/")
        self.coalition_url = (coalition_url or settings.coalition_url).rstrip("/")

    def dispatch_task(
        self, task: TaskContext, session_id: str | None = None, mode: str = "single"
    ) -> ModelContext:
        """Select agents and forward the ModelContext."""
        history: List[dict] = []
        memory: List[dict] = []
        if session_id:
            history = self._fetch_history(session_id)
            task.preferences = task.preferences or {}
            task.preferences["history"] = history
            if history:
                memory = history[-1].get("memory", [])

        TOKENS_IN.labels("task_dispatcher").inc(
            len(str(task.description or "").split())
        )

        agents = self._fetch_agents(task.task_type)
        ctx = ModelContext(
            task=task.task_id,
            task_context=task,
            session_id=session_id,
            memory=memory,
        )
        if mode == "single":
            agent = agents[0] if agents else None
            ctx.agent_selection = agent["id"] if agent else None
            if agent:
                arc = self._run_agent(agent, ctx)
                ctx.agents.append(arc)
                ctx.result = arc.result
        elif mode == "coalition":
            coalition = self._init_coalition(
                task.description or "", [a["id"] for a in agents]
            )
            task.preferences = task.preferences or {}
            task.preferences["coalition_id"] = coalition.get("id")
            for a in agents:
                self._assign_subtask(
                    coalition.get("id"), task.description or "", a["id"]
                )
            ctx.agents = [
                AgentRunContext(agent_id=a["id"], role=a.get("role"), url=a.get("url"))
                for a in agents
            ]
            ctx = self._send_to_coordinator(ctx, "parallel")
        else:
            ctx.agents = [
                AgentRunContext(agent_id=a["id"], role=a.get("role"), url=a.get("url"))
                for a in agents
            ]
            ctx = self._send_to_coordinator(ctx, mode)
        TASKS_PROCESSED.labels("task_dispatcher").inc()
        tokens = ctx.metrics.get("tokens_used", 0) if ctx.metrics else 0
        TOKENS_OUT.labels("task_dispatcher").inc(tokens)
        return ctx

    def _fetch_agents(self, capability: str) -> list[dict[str, Any]]:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.registry_url}/agents")
                resp.raise_for_status()
                agents = resp.json().get("agents", [])
                candidates = [
                    a
                    for a in agents
                    if capability in a.get("capabilities", [])
                    or capability in a.get("skills", [])
                ]
                candidates.sort(
                    key=lambda a: (
                        a.get("load_factor", 0),
                        a.get("estimated_cost_per_token", 0),
                        a.get("avg_response_time", 0),
                    )
                )
                return candidates
        except Exception:
            return []

    def _fetch_history(self, session_id: str) -> list[dict]:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.session_url}/context/{session_id}")
                resp.raise_for_status()
                return resp.json().get("context", [])
        except Exception:
            return []

    def _run_agent(self, agent: dict[str, Any], ctx: ModelContext) -> AgentRunContext:
        """Call the worker's /run endpoint and return AgentRunContext."""
        start = time.perf_counter()
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{agent['url'].rstrip('/')}/run",
                    json=ctx.model_dump(),
                    timeout=10,
                )
                resp.raise_for_status()
                data = ModelContext(**resp.json())
                arc = AgentRunContext(
                    agent_id=agent["id"],
                    role=agent.get("role"),
                    url=agent.get("url"),
                    result=data.result,
                    metrics=data.metrics,
                )
        except Exception:
            arc = AgentRunContext(
                agent_id=agent["id"], role=agent.get("role"), url=agent.get("url")
            )
        duration = time.perf_counter() - start
        self._update_status(agent["name"], duration)
        return arc

    def _send_to_coordinator(self, ctx: ModelContext, mode: str) -> ModelContext:
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.coordinator_url}/coordinate",
                    json={"context": ctx.model_dump(), "mode": mode},
                    timeout=10,
                )
                resp.raise_for_status()
                return ModelContext(**resp.json())
        except Exception:
            return ctx

    def _init_coalition(self, goal: str, members: List[str]) -> dict:
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.coalition_url}/coalition/init",
                    json={
                        "goal": goal,
                        "leader": members[0] if members else "",
                        "members": members,
                    },
                    timeout=5,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {
                "id": "local",
                "goal": goal,
                "leader": members[0] if members else "",
                "members": members,
                "strategy": "parallel-expert",
                "subtasks": [],
            }

    def _assign_subtask(self, coalition_id: str, title: str, assigned_to: str) -> None:
        try:
            with httpx.Client() as client:
                client.post(
                    f"{self.coalition_url}/coalition/{coalition_id}/assign",
                    json={"title": title, "assigned_to": assigned_to},
                    timeout=5,
                )
        except Exception:
            pass

    def _update_status(self, agent_name: str, duration: float) -> None:
        """Send runtime metrics to the registry."""
        payload = {
            "busy": False,
            "tasks_in_progress": 0,
            "last_response_duration": duration,
        }
        try:
            with httpx.Client() as client:
                client.post(
                    f"{self.registry_url}/agent_status/{agent_name}",
                    json=payload,
                    timeout=5,
                )
        except Exception:
            pass
