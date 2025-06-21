"""API routes for the Agent Registry service."""

from fastapi import APIRouter

from fastapi import HTTPException

from .schemas import AgentInfo, AgentList
from .service import AgentRegistryService

router = APIRouter()
service = AgentRegistryService()


@router.get("/agents", response_model=AgentList)
async def list_agents() -> AgentList:
    """Return all registered agents."""
    return AgentList(agents=service.list_agents())


@router.post("/register", response_model=AgentInfo)
async def register_agent(agent: AgentInfo) -> AgentInfo:
    """Register a new agent."""
    service.register_agent(agent)
    return agent


@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str) -> AgentInfo:
    """Return agent by id."""
    agent = service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
