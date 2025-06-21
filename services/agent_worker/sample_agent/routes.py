"""Routes for the sample Agent Worker."""

from fastapi import APIRouter

from core.model_context import ModelContext
from .service import SampleAgentService

router = APIRouter()
service = SampleAgentService()


@router.post("/run", response_model=ModelContext)
async def run(ctx: ModelContext) -> ModelContext:
    """Execute a task based on the provided ModelContext."""
    return service.run(ctx)
