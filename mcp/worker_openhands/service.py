"""Worker calling the OpenHands API for container operations."""

from __future__ import annotations

import json
import os
import asyncio
from typing import Any, Dict

from agents.openhands.docker_agent import DockerAgent


class WorkerService:
    """Interact with OpenHands API to perform container tasks."""

    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_OPENHANDS", "false").lower() == "true"
        api_url = os.getenv("OPENHANDS_API_URL", "http://openhands_api:8000")
        token = os.getenv("OPENHANDS_JWT")
        self.agent = DockerAgent(
            name="worker_openhands",
            api_url=api_url,
            github_token=token,
        )

    def _run_async(self, coro: asyncio.coroutines) -> Dict[str, Any]:
        return asyncio.run(coro)

    def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute docker related task via OpenHands."""
        if not self.enabled:
            return {
                "status": "disabled",
                "operation_id": None,
                "logs": "",
                "error": "OpenHands disabled",
            }

        async def run() -> Dict[str, Any]:
            await self.agent.initialize()
            try:
                payload = json.loads(task)
            except json.JSONDecodeError:
                payload = {"operation": "start_container", "image": task}

            op = payload.get("operation")

            if op == "start_container":
                result = await self.agent.run_container(
                    image=payload.get("image", "busybox"),
                    command=payload.get("command"),
                    environment=payload.get("environment"),
                    ports=payload.get("ports"),
                    volumes=payload.get("volumes"),
                )
                return {
                    "status": result.get("status", "running"),
                    "operation_id": result.get("container_id"),
                    "logs": result.get("logs", ""),
                    "error": None,
                }

            raise ValueError(f"Unsupported operation: {op}")

        try:
            return self._run_async(asyncio.wait_for(run(), timeout=30))
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "operation_id": None,
                "logs": "",
                "error": "timeout",
            }
        except Exception as exc:  # pragma: no cover - network errors
            return {
                "status": "error",
                "operation_id": None,
                "logs": "",
                "error": str(exc),
            }
