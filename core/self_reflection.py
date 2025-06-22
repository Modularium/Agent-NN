from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .agent_profile import AgentIdentity
from .feedback_loop import FeedbackLoopEntry


def reflect_and_adapt(
    agent: AgentIdentity, feedback_log: List[FeedbackLoopEntry]
) -> Dict[str, Any]:
    """Analyse feedback and return suggested profile adjustments."""
    counts = Counter(entry.event_type for entry in feedback_log)
    suggestions: Dict[str, Any] = {"traits": {}, "skills": [], "notes": []}
    if counts.get("task_failed", 0) >= 3:
        current = agent.traits.get("assertiveness", 1.0)
        suggestions["traits"]["assertiveness"] = max(0.0, current - 0.1)
        suggestions["notes"].append("reduced assertiveness")
    if counts.get("criticism_received", 0) >= 3:
        level = agent.traits.get("clarification", 0.0)
        suggestions["traits"]["clarification"] = level + 0.1
        suggestions["notes"].append("improved clarification")
    return suggestions
