from typing import List, Dict


class AgentRegistryService:
    """Stub registry storing available agents."""

    def __init__(self) -> None:
        self._agents: List[Dict[str, str]] = []

    def list_agents(self) -> List[Dict[str, str]]:
        return self._agents
