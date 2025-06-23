"""API routes for the LLM Gateway service."""

from fastapi import APIRouter

from core.model_context import ModelContext
from utils.api_utils import api_route

from .schemas import (
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
)
from .service import LLMGatewayService

router = APIRouter()
service = LLMGatewayService()


@api_route(version="v1.0.0")
@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    result = service.generate(req.prompt)
    return GenerateResponse(**result)


@api_route(version="v1.0.0")
@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    data = service.embed(req.text)
    return EmbedResponse(**data)


@api_route(version="v1.0.0")
@router.post("/chat", response_model=ChatResponse)
async def chat(ctx: ModelContext) -> ChatResponse:
    result = service.chat(ctx)
    return ChatResponse(**result)


@api_route(version="v1.0.0")
@router.get("/models", response_model=dict)
async def list_models() -> dict:
    return service.list_models()
