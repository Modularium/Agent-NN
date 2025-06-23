"""API routes for Routing Agent."""
from fastapi import APIRouter
from pydantic import BaseModel

from utils.api_utils import api_route
from .service import RoutingAgentService

router = APIRouter()
service = RoutingAgentService()


class RouteRequest(BaseModel):
    task_type: str
    required_tools: list[str] | None = None
    context: dict | None = None


@api_route(version="v1.0.0")
@router.post("/route")
async def route(req: RouteRequest) -> dict:
    """Return target worker for given task."""
    return service.route(req.task_type, req.required_tools, req.context)
