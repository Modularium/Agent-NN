"""High level HTTP client for Agent-NN services."""
from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

from ..config import SDKSettings


class AgentClient:
    """Wrapper for the Agent-NN REST API."""

    def __init__(self, settings: Optional[SDKSettings] = None) -> None:
        self.settings = settings or SDKSettings.load()
        self._client = httpx.Client(base_url=self.settings.host)

    def _headers(self) -> Dict[str, str]:
        if self.settings.api_token:
            return {"Authorization": f"Bearer {self.settings.api_token}"}
        return {}

    def submit_task(self, text: str) -> Dict[str, Any]:
        """Submit a task description and return the response."""
        resp = self._client.post("/task", json={"task": text}, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def list_agents(self) -> Dict[str, Any]:
        resp = self._client.get("/agents", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def list_sessions(self) -> Dict[str, Any]:
        resp = self._client.get("/sessions", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def vector_search(self, query: str, collection: str = "default") -> Dict[str, Any]:
        payload = {"query": query, "collection": collection}
        resp = self._client.post("/vector_search", json=payload, headers=self._headers())
        resp.raise_for_status()
        return resp.json()
