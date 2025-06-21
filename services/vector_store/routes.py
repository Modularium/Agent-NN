"""API routes for the Vector Store service."""

from fastapi import APIRouter
from utils.api_utils import api_route

from .schemas import (
    AddDocumentRequest,
    AddDocumentResponse,
    VectorSearchRequest,
    VectorSearchResponse,
)
from .service import VectorStoreService

router = APIRouter()
service = VectorStoreService()


@api_route(version="v1.0.0")
@router.post("/add_document", response_model=AddDocumentResponse)
async def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    """Add a new document to a collection."""
    doc_id = service.add_document(req.text, req.collection)
    return AddDocumentResponse(id=doc_id)


@api_route(version="v1.0.0")
@router.post("/vector_search", response_model=VectorSearchResponse)
async def vector_search(req: VectorSearchRequest) -> VectorSearchResponse:
    """Perform a vector search over the knowledge base."""
    data = service.search(req.query, req.collection, top_k=req.top_k)
    return VectorSearchResponse(**data)
