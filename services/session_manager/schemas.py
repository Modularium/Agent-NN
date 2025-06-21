"""Data models for Session Manager."""

from typing import Dict

from pydantic import BaseModel


class SessionData(BaseModel):
    data: Dict
