"""API routes for the Agent Coordinator service."""

from fastapi import APIRouter
from utils.api_utils import api_route

from core.model_context import ModelContext
from .service import AgentCoordinatorService

router = APIRouter()
service = AgentCoordinatorService()


@api_route(version="v1.0.0")
@router.post("/coordinate", response_model=ModelContext)
async def coordinate(data: dict) -> ModelContext:
    ctx = ModelContext(**data.get("context"))
    mode = data.get("mode", "parallel")
    return service.coordinate(ctx, mode=mode)
