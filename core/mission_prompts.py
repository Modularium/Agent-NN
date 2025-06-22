from __future__ import annotations

from typing import Any, Dict

from .agent_profile import AgentIdentity


def render_prompt(agent: AgentIdentity, step: Dict[str, Any], team_context: Dict[str, Any]) -> str:
    """Return a prompt string for ``agent`` based on mission ``step``."""
    role = step.get("role", agent.role)
    task = step.get("task", "")
    goal = team_context.get("goal", "")
    return f"Role: {role}\nGoal: {goal}\nTask: {task}"
