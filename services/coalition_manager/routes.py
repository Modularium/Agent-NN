"""API routes for the Coalition Manager service."""
from __future__ import annotations

from dataclasses import asdict
from fastapi import APIRouter
from utils.api_utils import api_route

from .schemas import CoalitionInit, SubtaskAssign, CoalitionData
from .service import CoalitionManagerService

router = APIRouter()
service = CoalitionManagerService()


@api_route(version="v1.0.0")
@router.post("/coalition/init", response_model=CoalitionData)
async def init_coalition(data: CoalitionInit) -> CoalitionData:
    coalition = service.create_coalition(
        data.goal, data.leader, data.members, data.strategy
    )
    return CoalitionData(**asdict(coalition))


@api_route(version="v1.0.0")
@router.post("/coalition/{coalition_id}/assign", response_model=CoalitionData)
async def assign_subtask(coalition_id: str, data: SubtaskAssign) -> CoalitionData:
    coalition = service.assign_subtask(coalition_id, data.title, data.assigned_to)
    return CoalitionData(**asdict(coalition))


@api_route(version="v1.0.0")
@router.get("/coalition/{coalition_id}", response_model=CoalitionData)
async def get_coalition(coalition_id: str) -> CoalitionData:
    coalition = service.get_coalition(coalition_id)
    return CoalitionData(**asdict(coalition))
