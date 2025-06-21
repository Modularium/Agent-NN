"""Simple critic agent evaluating text proposals."""

from __future__ import annotations

from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT


class CriticAgent:
    """Provide a numeric score and feedback for a given text."""

    def vote(self, text: str, criteria: str = "", context: dict | None = None) -> dict:
        tokens = len(text.split())
        TOKENS_IN.labels("critic_agent").inc(tokens)
        # very naive scoring based on length
        score = min(1.0, tokens / 50)
        feedback = "Well structured" if score > 0.5 else "Too short"
        TOKENS_OUT.labels("critic_agent").inc(0)
        TASKS_PROCESSED.labels("critic_agent").inc()
        return {"score": score, "feedback": feedback}
