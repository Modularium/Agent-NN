"""Routes for the Example Agent Worker."""

from fastapi import APIRouter
from utils.api_utils import api_route

from .schemas import ExecuteRequest, ExecuteResponse
from .service import ExampleAgentService

router = APIRouter()
service = ExampleAgentService()


@api_route(version="dev")  # \U0001F6A7 experimental
@router.post("/execute_task", response_model=ExecuteResponse)
async def execute_task(req: ExecuteRequest) -> ExecuteResponse:
    """Execute a task and return the result."""
    return ExecuteResponse(**service.execute_task(req.task))
