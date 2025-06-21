"""Task dispatcher core logic."""

import random
from typing import Any, List

import httpx

from core.model_context import ModelContext, TaskContext
from .config import settings


class TaskDispatcherService:
    """Dispatch incoming tasks to worker agents."""

    def __init__(self, registry_url: str | None = None, session_url: str | None = None) -> None:
        self.registry_url = (registry_url or settings.registry_url).rstrip("/")
        self.session_url = (session_url or settings.session_url).rstrip("/")

    def dispatch_task(self, task: TaskContext, session_id: str | None = None) -> ModelContext:
        """Select an agent and forward the ModelContext to it."""
        history: List[dict] = []
        if session_id:
            history = self._fetch_history(session_id)
            task.preferences = task.preferences or {}
            task.preferences["history"] = history

        agents = self._fetch_agents(task.task_type)
        agent = random.choice(agents) if agents else None
        ctx = ModelContext(
            task=task.task_id,
            task_context=task,
            agent_selection=agent["id"] if agent else None,
            session_id=session_id,
        )
        if agent:
            ctx = self._send_to_worker(agent, ctx)
        return ctx

    def _fetch_agents(self, capability: str) -> list[dict[str, Any]]:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.registry_url}/agents")
                resp.raise_for_status()
                agents = resp.json().get("agents", [])
                return [a for a in agents if capability in a.get("capabilities", [])]
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

    def _send_to_worker(self, agent: dict[str, Any], ctx: ModelContext) -> ModelContext:
        """Call the worker's /run endpoint with the context."""
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{agent['url'].rstrip('/')}/run",
                    json=ctx.model_dump(),
                    timeout=10,
                )
                resp.raise_for_status()
                return ModelContext(**resp.json())
        except Exception:
            return ctx
