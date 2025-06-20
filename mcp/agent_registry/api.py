from fastapi import APIRouter

from .service import AgentRegistryService

router = APIRouter()
service = AgentRegistryService()


@router.get("/agents")
async def get_agents() -> list:
    """Return registered agents."""
    return service.list_agents()


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
