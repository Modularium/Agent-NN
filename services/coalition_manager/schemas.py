"""Data models for the Coalition Manager."""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class CoalitionInit(BaseModel):
    goal: str
    leader: str
    members: List[str] = Field(default_factory=list)
    strategy: str = "plan-then-split"


class SubtaskAssign(BaseModel):
    title: str
    assigned_to: str


class CoalitionData(BaseModel):
    id: str
    goal: str
    leader: str
    members: List[str]
    strategy: str
    subtasks: List[Dict[str, Any]]
