"""Routes for the Example Agent Worker."""

from fastapi import APIRouter

from .schemas import ExecuteRequest, ExecuteResponse
from .service import ExampleAgentService

router = APIRouter()
service = ExampleAgentService()


@router.post("/execute_task", response_model=ExecuteResponse)
async def execute_task(req: ExecuteRequest) -> ExecuteResponse:
    """Execute a task and return the result."""
    return ExecuteResponse(**service.execute_task(req.task))
