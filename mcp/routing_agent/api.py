from fastapi import APIRouter
from pydantic import BaseModel

from .service import RoutingAgentService

router = APIRouter()
service = RoutingAgentService()


class RouteRequest(BaseModel):
    task_type: str
    required_tools: list[str] | None = None
    context: dict | None = None


@router.post("/route")
async def route(req: RouteRequest) -> dict:
    return service.route(req.task_type, req.required_tools, req.context)


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
