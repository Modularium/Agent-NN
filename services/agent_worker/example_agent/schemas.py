"""Schemas for Example Agent Worker."""

from pydantic import BaseModel


class ExecuteRequest(BaseModel):
    task: str


class ExecuteResponse(BaseModel):
    result: str
