"""Data models for the Agent Registry API."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from uuid import uuid4


class AgentInfo(BaseModel):
    """Information about a registered agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    url: str
    domain: str | None = None
    version: str | None = None
    capabilities: list[str] = []
    role: str | None = None
    traits: Dict[str, Any] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    estimated_cost_per_token: float = 0.0
    avg_response_time: float = 0.0
    load_factor: float = 0.0


class AgentList(BaseModel):
    agents: List[AgentInfo]
