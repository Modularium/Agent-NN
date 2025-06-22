"""Service for the critic agent."""
from __future__ import annotations

from services.agent_worker.demo_agents.critic_agent import CriticAgent


class CriticAgentService:
    """Wrap the critic agent for HTTP usage."""

    def __init__(self) -> None:
        self.agent = CriticAgent()

    def vote(self, text: str, criteria: str, context: dict | None = None) -> dict:
        return self.agent.vote(text, criteria, context)
