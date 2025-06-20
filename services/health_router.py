from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health")
async def health() -> dict:
    """Simple health check."""
    return {"status": "ok"}


@health_router.get("/status")
async def status() -> dict:
    """Return basic running status."""
    return {"status": "running"}
