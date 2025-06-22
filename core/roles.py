from enum import Enum


class AgentRole(str, Enum):
    """Predefined roles for agent authorization."""

    WRITER = "writer"
    RETRIEVER = "retriever"
    CRITIC = "critic"
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
