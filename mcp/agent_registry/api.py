from fastapi import APIRouter
from pydantic import BaseModel

from .service import AgentRegistryService

router = APIRouter()
service = AgentRegistryService()


@router.get("/agents")
async def get_agents() -> list:
    """Return registered agents."""
    return service.list_agents()


class AgentRegistration(BaseModel):
    name: str
    agent_type: str
    url: str
    capabilities: list[str]
    status: str = "online"


@router.post("/register")
async def register_agent(agent: AgentRegistration) -> dict:
    service.register_agent(agent.dict())
    return {"status": "registered"}


@router.get("/agents/health")
async def agents_health() -> list:
    return service.health_status()


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
