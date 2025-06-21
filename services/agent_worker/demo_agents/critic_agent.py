"""Simple critic agent evaluating text proposals."""

from __future__ import annotations

from datetime import datetime

from core.agent_bus import publish, subscribe
from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT
from core.agent_profile import AgentIdentity


class CriticAgent:
    """Provide a numeric score and feedback for a given text."""

    def __init__(self) -> None:
        self.profile = AgentIdentity.load("critic_agent")
        self._ratings = self.profile.traits.get("ratings", 0)
        self._avg_dev = self.profile.traits.get("avg_deviation", 0.0)

    def vote(self, text: str, criteria: str = "", context: dict | None = None) -> dict:
        tokens = len(text.split())
        TOKENS_IN.labels("critic_agent").inc(tokens)
        score = min(1.0, tokens / 50)
        feedback = "Well structured" if score > 0.5 else "Too short"
        TOKENS_OUT.labels("critic_agent").inc(0)
        TASKS_PROCESSED.labels("critic_agent").inc()

        # update statistics
        self._avg_dev = ((self._avg_dev * self._ratings) + abs(1.0 - score)) / (self._ratings + 1)
        self._ratings += 1
        self.profile.traits["avg_deviation"] = self._avg_dev
        self.profile.traits["ratings"] = self._ratings
        self.profile.save()

        return {"score": score, "feedback": feedback}

    def process_bus(self) -> None:
        """Handle queued requests via AgentBus."""
        for msg in subscribe("critic_agent"):
            if msg.get("type") == "request":
                text = msg.get("payload", {}).get("text", "")
                result = self.vote(text)
                publish(
                    msg["sender"],
                    {
                        "sender": "critic_agent",
                        "receiver": msg["sender"],
                        "type": "feedback",
                        "payload": result,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

