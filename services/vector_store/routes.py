"""API routes for the Vector Store service."""

from fastapi import APIRouter

from .schemas import VectorSearchRequest, VectorSearchResponse
from .service import VectorStoreService

router = APIRouter()
service = VectorStoreService()


@router.post("/vector_search", response_model=VectorSearchResponse)
async def vector_search(req: VectorSearchRequest) -> VectorSearchResponse:
    """Perform a vector search over the knowledge base."""
    return VectorSearchResponse(**service.search(req.text, req.embedding_model))
