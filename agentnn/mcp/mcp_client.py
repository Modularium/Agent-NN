"""HTTP client wrapper for MCP compatible services."""

from __future__ import annotations

from typing import Any, List

import httpx

from core.model_context import ModelContext

from .context_adapter import from_mcp, to_mcp


class MCPClient:
    """Simple MCP client for executing tasks and managing context."""

    def __init__(self, endpoint: str = "http://localhost:9000") -> None:
        self._client = httpx.Client(base_url=endpoint.rstrip("/"))

    def execute(self, ctx: ModelContext) -> ModelContext:
        """Dispatch a context to an MCP server and return the updated context."""
        resp = self._client.post("/v1/mcp/execute", json=to_mcp(ctx))
        resp.raise_for_status()
        return from_mcp(resp.json())

    def update_context(self, ctx: ModelContext) -> dict[str, Any]:
        """Store context information at the MCP server."""
        resp = self._client.post("/v1/mcp/context", json=to_mcp(ctx))
        resp.raise_for_status()
        return resp.json()

    def get_context(self, session_id: str) -> List[ModelContext]:
        """Fetch all contexts for a given session."""
        resp = self._client.get(f"/v1/mcp/context/{session_id}")
        resp.raise_for_status()
        data = resp.json().get("context", [])
        return [from_mcp(item) for item in data]
