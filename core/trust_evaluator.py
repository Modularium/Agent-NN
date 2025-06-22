from __future__ import annotations

from typing import Any, Dict, List

from .governance import AgentContract
from .roles import resolve_roles


def calculate_trust(agent_id: str, context: List[Dict[str, Any]]) -> float:
    """Return trust score between 0.0 and 1.0 based on past context."""
    if not context:
        return 0.0

    success = 0
    total = 0
    feedback_sum = 0.0
    efficiency_sum = 0.0
    reliability_sum = 0.0

    for entry in context:
        if entry.get("agent_id") != agent_id:
            continue
        total += 1
        if entry.get("success", True):
            success += 1
        feedback_sum += float(entry.get("feedback_score", 0.0))
        tokens_used = float(entry.get("metrics", {}).get("tokens_used", 1))
        expected = float(entry.get("expected_tokens", tokens_used))
        efficiency_sum += expected / tokens_used if tokens_used else 1.0
        reliability_sum += 1.0 if not entry.get("error") else 0.0

    if total == 0:
        return 0.0

    success_rate = success / total
    feedback_score = feedback_sum / total
    token_efficiency = efficiency_sum / total
    reliability = reliability_sum / total
    trust = (success_rate + feedback_score + token_efficiency + reliability) / 4
    return max(0.0, min(1.0, trust))


def eligible_for_role(agent_id: str, target_role: str) -> bool:
    """Return True if the agent qualifies for ``target_role``."""

    contract = AgentContract.load(agent_id)
    history = contract.constraints.get("task_history", [])
    trust = calculate_trust(agent_id, history)
    standing = float(contract.constraints.get("standing", 1.0))
    if target_role in resolve_roles(agent_id):
        return False
    score = trust * standing
    return score >= contract.trust_level_required and len(history) >= 5


def update_trust_usage(agent_id: str, tokens_used: int, limit: int) -> None:
    """Adjust trust_score based on token usage."""
    contract = AgentContract.load(agent_id)
    score = float(contract.constraints.get("trust_score", 1.0))
    ratio = tokens_used / limit if limit else 0.0
    if ratio and ratio <= 0.8:
        score = min(1.0, score + 0.05)
    elif ratio and ratio > 1.0:
        score = max(0.0, score - 0.1)
        if score < contract.trust_level_required and contract.allowed_roles:
            contract.allowed_roles = contract.allowed_roles[:1]
    contract.constraints["trust_score"] = score
    contract.save()
