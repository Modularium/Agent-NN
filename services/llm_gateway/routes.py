"""API routes for the LLM Gateway service."""

from fastapi import APIRouter

from .schemas import GenerateRequest, GenerateResponse
from .service import LLMGatewayService

router = APIRouter()
service = LLMGatewayService()


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text based on a prompt."""
    return GenerateResponse(**service.generate(req.prompt))
