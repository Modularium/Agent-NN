from fastapi import APIRouter
from pydantic import BaseModel

from .service import LLMGatewayService

router = APIRouter()
service = LLMGatewayService()


class GenerateRequest(BaseModel):
    prompt: str


@router.post("/generate")
async def generate(req: GenerateRequest) -> dict:
    text = service.generate(req.prompt)
    return {"text": text}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
