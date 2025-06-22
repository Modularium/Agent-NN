from __future__ import annotations

from typing import Any, Dict, List


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
