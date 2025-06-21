"""API routes for the Task Dispatcher service."""

from fastapi import APIRouter

from .schemas import TaskRequest
from core.model_context import ModelContext
from .service import TaskDispatcherService

router = APIRouter()
service = TaskDispatcherService()


@router.post("/task", response_model=ModelContext)
async def create_task(task: TaskRequest) -> ModelContext:
    """Accept a TaskContext and queue it for processing."""
    return service.dispatch_task(task)
