"""In-memory session manager supporting multiple agents per session."""

from __future__ import annotations

from typing import Any, Dict
import uuid

from core.model_context import ModelContext, TaskContext
from ..mcp.mcp_client import MCPClient
from ..mcp.mcp_ws import ws_server


class SessionManager:
    """Manage dialog sessions with a pool of agents."""

    def __init__(self, endpoint: str = "http://localhost:8090") -> None:
        self.client = MCPClient(endpoint)
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self) -> str:
        """Return a new session id."""
        sid = str(uuid.uuid4())
        self._sessions[sid] = {"linked_agents": [], "message_history": []}
        try:
            # broadcast creation event
            ws_server.broadcast({"event": "session_created", "session_id": sid})
        except Exception:
            pass
        return sid

    def add_agent(self, session_id: str, agent_id: str) -> None:
        """Attach an agent to the session."""
        session = self._sessions.setdefault(
            session_id, {"linked_agents": [], "message_history": []}
        )
        session["linked_agents"].append(agent_id)
        try:
            ws_server.broadcast(
                {"event": "agent_added", "session_id": session_id, "agent": agent_id}
            )
        except Exception:
            pass

    def run_task(self, session_id: str, task: str) -> ModelContext:
        """Execute the task with all linked agents."""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("unknown session")
        result_ctx: ModelContext | None = None
        for agent in session["linked_agents"]:
            ctx = ModelContext(
                session_id=session_id,
                task_context=TaskContext(task_type="chat", description=task),
                agent_selection=agent,
            )
            result_ctx = self.client.execute(ctx)
            session["message_history"].append(
                {"agent": agent, "task": task, "result": result_ctx.result}
            )
            try:
                ws_server.broadcast(
                    {
                        "event": "agent_result",
                        "session_id": session_id,
                        "agent": agent,
                        "result": result_ctx.result,
                    }
                )
            except Exception:
                pass
        return result_ctx or ModelContext(task_context=TaskContext(task_type="chat", description=task))

    def get_session(self, session_id: str) -> Dict[str, Any] | None:
        """Return stored metadata for a session."""
        return self._sessions.get(session_id)
