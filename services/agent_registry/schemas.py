"""Data models for the Agent Registry API."""

from typing import List

from pydantic import BaseModel


class AgentInfo(BaseModel):
    name: str
    url: str


class AgentList(BaseModel):
    agents: List[AgentInfo]
