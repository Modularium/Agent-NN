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


class TranslateRequest(BaseModel):
    text: str
    target_lang: str


class VisionRequest(BaseModel):
    image_url: str


@router.post("/chain/qa")
async def chain_qa(req: QARequest) -> dict:
    """Run a retrieval augmented generation chain."""
    text = service.qa(req.query)
    return {"text": text}


@router.post("/translate")
async def translate(req: TranslateRequest) -> dict:
    """Translate text into the target language."""
    text = service.translate(req.text, req.target_lang)
    return {"text": text}


@router.post("/vision")
async def vision(req: VisionRequest) -> dict:
    """Describe an image from a URL. Placeholder for multimodal models."""
    text = service.vision_describe(req.image_url)
    return {"text": text}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
