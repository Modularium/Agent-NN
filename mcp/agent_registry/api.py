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


class PluginRegistration(BaseModel):
    name: str
    description: str


@router.post("/register")
async def register_agent(agent: AgentRegistration) -> dict:
    service.register_agent(agent.dict())
    return {"status": "registered"}


@router.post("/register_plugin")
async def register_plugin(plugin: PluginRegistration) -> dict:
    service.register_plugin(plugin.dict())
    return {"status": "registered"}


@router.get("/agents/health")
async def agents_health() -> list:
    return service.health_status()


@router.get("/plugins")
async def get_plugins() -> list:
    return service.list_plugins()


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
