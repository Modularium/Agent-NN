"""API routes for the Task Dispatcher service."""

from fastapi import APIRouter

from .schemas import TaskRequest
from .service import TaskDispatcherService

router = APIRouter()
service = TaskDispatcherService()


@router.post("/task")
async def create_task(task: TaskRequest) -> dict:
    """Accept a TaskContext and queue it for processing."""
    return service.dispatch_task(task)
