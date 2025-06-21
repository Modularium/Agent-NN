"""Task dispatcher core logic."""

import random
from typing import Any

import httpx

from core.model_context import ModelContext, TaskContext
from .config import settings


class TaskDispatcherService:
    """Dispatch incoming tasks to worker agents."""

    def dispatch_task(self, task: TaskContext) -> ModelContext:
        """Select an agent and forward the ModelContext to it."""
        agents = self._fetch_agents(task.task_type)
        agent = random.choice(agents) if agents else None
        ctx = ModelContext(
            task=task.task_id,
            task_context=task,
            agent_selection=agent["id"] if agent else None,
        )
        if agent:
            ctx = self._send_to_worker(agent, ctx)
        return ctx

    def _fetch_agents(self, capability: str) -> list[dict[str, Any]]:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{settings.registry_url}/agents")
                resp.raise_for_status()
                agents = resp.json().get("agents", [])
                return [a for a in agents if capability in a.get("capabilities", [])]
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
