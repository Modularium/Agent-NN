"""API routes for the Agent Registry service."""

from fastapi import APIRouter
from utils.api_utils import api_route

from fastapi import HTTPException

from dataclasses import asdict
from .schemas import AgentInfo, AgentList
from core.agent_profile import AgentIdentity
from .service import AgentRegistryService

router = APIRouter()
service = AgentRegistryService()


@api_route(version="v1.0.0")
@router.get("/agents", response_model=AgentList)
async def list_agents() -> AgentList:
    """Return all registered agents with profile summary."""
    agents = []
    for info in service.list_agents():
        profile = AgentIdentity.load(info.name)
        info.role = profile.role or info.role
        info.traits = {"summary": profile.traits.get("summary", "")}
        info.skills = profile.skills[:3]
        info.estimated_cost_per_token = profile.estimated_cost_per_token
        info.avg_response_time = profile.avg_response_time
        info.load_factor = profile.load_factor
        agents.append(info)
    return AgentList(agents=agents)


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


@api_route(version="v1.0.0")
@router.get("/agent_profile/{agent_name}")
async def get_agent_profile(agent_name: str) -> dict:
    """Return stored profile for an agent."""
    profile = AgentIdentity.load(agent_name)
    return asdict(profile)


@api_route(version="v1.0.0")
@router.post("/agent_profile/{agent_name}")
async def update_agent_profile(agent_name: str, data: dict) -> dict:
    """Update profile traits or skills."""
    profile = AgentIdentity.load(agent_name)
    traits = data.get("traits")
    if isinstance(traits, dict):
        profile.traits.update(traits)
    skills = data.get("skills")
    if isinstance(skills, list):
        profile.skills = skills
    profile.save()
    return asdict(profile)


@api_route(version="v1.0.0")
@router.post("/agent_status/{agent_name}")
async def update_agent_status(agent_name: str, data: dict) -> dict:
    """Update runtime status for an agent."""
    service.update_status(agent_name, data)
    profile = AgentIdentity.load(agent_name)
    profile.update_metrics(
        response_time=data.get("last_response_duration"),
        tasks_in_progress=data.get("tasks_in_progress"),
    )
    return {"status": "ok"}


@api_route(version="v1.0.0")
@router.get("/agent_status/{agent_name}")
async def get_agent_status(agent_name: str) -> dict:
    """Return current runtime status for an agent."""
    status = service.get_status(agent_name) or {}
    profile = AgentIdentity.load(agent_name)
    status.update(
        {
            "estimated_cost_per_token": profile.estimated_cost_per_token,
            "avg_response_time": profile.avg_response_time,
            "load_factor": profile.load_factor,
        }
    )
    return status
