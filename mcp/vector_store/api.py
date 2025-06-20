from fastapi import APIRouter
from pydantic import BaseModel

from .service import VectorStoreService

router = APIRouter()
service = VectorStoreService()


class QueryRequest(BaseModel):
    query: str


class DocumentRequest(BaseModel):
    document: dict


@router.post("/query")
async def query(req: QueryRequest) -> list:
    return service.query(req.query)


@router.post("/document")
async def add_document(req: DocumentRequest) -> dict:
    return service.add_document(req.document)


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
