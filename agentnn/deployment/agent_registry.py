"""Client helpers for API based agent deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import httpx
import yaml


class AgentRegistry:
    """Interact with the MCP agent deployment API."""

    def __init__(self, endpoint: str = "http://localhost:8090") -> None:
        self._client = httpx.Client(base_url=endpoint.rstrip("/"))
        self._prefix = "/v1/mcp/agent"

    def deploy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy an agent based on the given config."""
        resp = self._client.post(f"{self._prefix}/deploy", json=config)
        resp.raise_for_status()
        return resp.json()

    def remove(self, name: str) -> Dict[str, Any]:
        """Remove an agent from the registry."""
        resp = self._client.post(f"{self._prefix}/remove", json={"name": name})
        resp.raise_for_status()
        return resp.json()

    def list_agents(self) -> List[Dict[str, Any]]:
        """Return all registered agents."""
        resp = self._client.get(f"{self._prefix}/list")
        resp.raise_for_status()
        return resp.json().get("agents", [])


def load_agent_file(path: Path) -> Dict[str, Any]:
    """Return agent configuration from YAML file."""
    data = yaml.safe_load(path.read_text())
    return data or {}

