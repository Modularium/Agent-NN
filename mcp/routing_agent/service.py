"""Simple routing agent using rule files."""

from __future__ import annotations

import os
from typing import Dict, List

import yaml


class RoutingAgentService:
    """Route tasks to workers or tool chains based on rules."""

    def __init__(self, rules_path: str = "mcp/routing_agent/rules.yaml") -> None:
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as fh:
                self.rules: Dict[str, str] = yaml.safe_load(fh) or {}
        else:
            self.rules = {}

    def route(
        self,
        task_type: str,
        required_tools: List[str] | None = None,
        context: Dict | None = None,
    ) -> Dict:
        target = self.rules.get(task_type)
        return {"target": target or "unknown"}
