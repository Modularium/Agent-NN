"""Manage agent coalitions."""
from __future__ import annotations

from typing import List, Dict, Any

from core.coalitions import AgentCoalition, create_coalition


class CoalitionManagerService:
    """Create and update agent coalitions."""

    def create_coalition(
        self,
        goal: str,
        leader: str,
        members: List[str] | None = None,
        strategy: str = "plan-then-split",
    ) -> AgentCoalition:
        coalition = create_coalition(goal, leader, members, strategy)
        return coalition

    def assign_subtask(
        self, coalition_id: str, title: str, assigned_to: str
    ) -> AgentCoalition:
        coalition = AgentCoalition.load(coalition_id)
        coalition.subtasks.append({
            "title": title,
            "assigned_to": assigned_to,
            "status": "pending",
        })
        coalition.save()
        return coalition

    def get_coalition(self, coalition_id: str) -> AgentCoalition:
        return AgentCoalition.load(coalition_id)
