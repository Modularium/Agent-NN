import json
import os
import urllib.request
from typing import Dict, List

import yaml


class AgentRegistryService:
    """Simple in-memory registry for available agents."""

    def __init__(self, config_path: str = "mcp/agents.yaml") -> None:
        self._agents: Dict[str, Dict[str, str]] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
                for agent in cfg.get("agents", []):
                    self._agents[agent["name"]] = agent

    def list_agents(self) -> List[Dict[str, str]]:
        return list(self._agents.values())

    def register_agent(self, info: Dict[str, str]) -> None:
        self._agents[info["name"]] = info

    def health_status(self) -> List[Dict[str, str]]:
        statuses: List[Dict[str, str]] = []
        for agent in self._agents.values():
            health_url = agent["url"].rstrip("/") + "/health"
            status = "unknown"
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    status = data.get("status", "unknown")
            except Exception:
                status = "unreachable"
            statuses.append({"name": agent["name"], "status": status})
        return statuses
