"""API routes for the Federation Manager service."""

from fastapi import APIRouter

from utils.api_utils import api_route
from core.model_context import ModelContext
from .service import FederationManagerService

router = APIRouter()
service = FederationManagerService()


@api_route(version="v1.0.0")
@router.post("/nodes")
async def register(name: str, base_url: str) -> dict:
    service.register_node(name, base_url)
    return {"status": "ok"}


@api_route(version="v1.0.0")
@router.post("/nodes/{name}/heartbeat")
async def heartbeat(name: str) -> dict:
    service.heartbeat(name)
    return {"status": "ok"}


@api_route(version="v1.0.0")
@router.get("/nodes")
async def list_nodes() -> dict:
    return {"nodes": list(service.list_nodes().keys())}


@api_route(version="v1.0.0")
@router.delete("/nodes/{name}")
async def remove(name: str) -> dict:
    service.remove_node(name)
    return {"status": "ok"}


@api_route(version="v1.0.0")
@router.post("/dispatch/{name}", response_model=ModelContext)
async def dispatch(name: str, ctx: ModelContext) -> ModelContext:
    target = None if name == "auto" else name
    return service.dispatch(target, ctx)
