"""API routes for the Agent Registry service."""

from fastapi import APIRouter
from utils.api_utils import api_route

from fastapi import HTTPException

from .schemas import AgentInfo, AgentList
from .service import AgentRegistryService

router = APIRouter()
service = AgentRegistryService()


@api_route(version="v1.0.0")
@router.get("/agents", response_model=AgentList)
async def list_agents() -> AgentList:
    """Return all registered agents."""
    return AgentList(agents=service.list_agents())


@api_route(version="v1.0.0")
@router.post("/register", response_model=AgentInfo)
async def register_agent(agent: AgentInfo) -> AgentInfo:
    """Register a new agent."""
    service.register_agent(agent)
    return agent


@api_route(version="v1.0.0")
@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str) -> AgentInfo:
    """Return agent by id."""
    agent = service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
