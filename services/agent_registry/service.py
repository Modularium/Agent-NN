"""Agent registry logic."""

from typing import Dict, List


class AgentRegistryService:
    """Maintain a list of available agents."""

    def __init__(self) -> None:
        self._agents: List[Dict[str, str]] = []

    def list_agents(self) -> List[Dict[str, str]]:
        """Return registered agents."""
        return self._agents
