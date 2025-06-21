"""API routes for the critic agent."""
from fastapi import APIRouter
from utils.api_utils import api_route

from .service import CriticAgentService

router = APIRouter()
service = CriticAgentService()


@api_route(version="v1.0.0")
@router.post("/vote")
async def vote(data: dict) -> dict:
    text = data.get("text", "")
    criteria = data.get("criteria", "")
    context = data.get("context")
    return service.vote(text, criteria, context)
