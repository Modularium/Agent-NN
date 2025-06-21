from fastapi import APIRouter
from utils.api_utils import api_route

health_router = APIRouter()


@api_route(version="v1.0.0")
@health_router.get("/health")
async def health() -> dict:
    """Simple health check."""
    return {"status": "ok"}


@api_route(version="v1.0.0")
@health_router.get("/status")
async def status() -> dict:
    """Return basic running status."""
    return {"status": "running"}
