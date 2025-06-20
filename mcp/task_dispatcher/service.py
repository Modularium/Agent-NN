import json
import urllib.request
import logging
from typing import Dict, Optional

from mcp.agent_registry.service import AgentRegistryService
from mcp.session_manager.service import SessionManagerService


class TaskDispatcherService:
    """Dispatch tasks to registered worker services."""

    def __init__(self) -> None:
        self.registry = AgentRegistryService()
        self.sessions = SessionManagerService()
        # simple rule mapping task_type to agent name
        self.task_map = {"greeting": "worker_dev"}
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def dispatch_task(
        self, task_type: str, task_input: str, session_id: Optional[str]
    ) -> Dict[str, str]:
        """Delegate a task to a worker and return its result."""
        session_data = self.sessions.get_session(session_id) if session_id else None

        agent_name = self.task_map.get(task_type)
        agent = next(
            (a for a in self.registry.list_agents() if a["name"] == agent_name), None
        )
        if not agent:
            return {"error": "no agent for task"}

        url = agent["url"].rstrip("/") + "/execute_task"
        payload = json.dumps({"task": task_input}).encode()
        try:
            self.logger.info("Dispatch %s to %s", task_type, url)
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception as exc:
            self.logger.error("Worker call failed: %s", exc)
            result = {"error": str(exc)}

        if session_id:
            history = session_data.get("history", []) if session_data else []
            history.append(
                {"task_type": task_type, "input": task_input, "result": result}
            )
            self.sessions.update_session(session_id, {"history": history})

        return {"worker": agent["name"], "response": result}
