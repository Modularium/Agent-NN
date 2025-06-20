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


class QARequest(BaseModel):
    query: str


@router.post("/chain/qa")
async def chain_qa(req: QARequest) -> dict:
    """Run a retrieval augmented generation chain."""
    text = service.qa(req.query)
    return {"text": text}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
