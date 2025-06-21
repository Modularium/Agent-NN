"""API routes for the Agent Registry service."""

from fastapi import APIRouter

from .schemas import AgentList
from .service import AgentRegistryService

router = APIRouter()
service = AgentRegistryService()


@router.get("/agents", response_model=AgentList)
async def list_agents() -> AgentList:
    """Return all registered agents."""
    return AgentList(agents=service.list_agents())
