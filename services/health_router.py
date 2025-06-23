from fastapi import APIRouter

from core.schemas import StatusResponse
from utils.api_utils import api_route

health_router = APIRouter()


@api_route(version="v1.0.0")
@health_router.get("/health", response_model=StatusResponse)
async def health() -> StatusResponse:
    """Simple health check."""
    return StatusResponse(status="ok")


@api_route(version="v1.0.0")
@health_router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Return basic running status."""
    return StatusResponse(status="running")
