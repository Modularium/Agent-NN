"""API routes for the LLM Gateway service."""

from fastapi import APIRouter

from .schemas import (
    GenerateRequest,
    GenerateResponse,
    EmbedRequest,
    EmbedResponse,
)
from .service import LLMGatewayService

router = APIRouter()
service = LLMGatewayService()


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text based on a prompt."""
    result = service.generate(
        req.prompt, model_name=req.model_name, temperature=req.temperature or 0.7
    )
    return GenerateResponse(**result)


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """Return an embedding for the given text."""
    data = service.embed(req.text, model_name=req.model_name)
    return EmbedResponse(**data)
