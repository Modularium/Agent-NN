"""Routes for the sample Agent Worker."""

from fastapi import APIRouter
from utils.api_utils import api_route

from core.model_context import ModelContext
from .service import SampleAgentService

router = APIRouter()
service = SampleAgentService()


@api_route(version="dev")  # \U0001F6A7 experimental
@router.post("/run", response_model=ModelContext)
async def run(ctx: ModelContext) -> ModelContext:
    """Execute a task based on the provided ModelContext."""
    return service.run(ctx)
