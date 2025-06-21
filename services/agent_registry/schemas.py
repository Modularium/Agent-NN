"""Data models for the Agent Registry API."""

from typing import List

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


class AgentList(BaseModel):
    agents: List[AgentInfo]
