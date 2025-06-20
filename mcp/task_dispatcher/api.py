from fastapi import APIRouter
from pydantic import BaseModel

from .service import TaskDispatcherService

router = APIRouter()
service = TaskDispatcherService()


class TaskRequest(BaseModel):
    task_type: str
    input: str
    session_id: str | None = None


@router.post("/task")
async def create_task(request: TaskRequest):
    """Accept a new task and dispatch it."""
    return service.dispatch_task(request.task_type, request.input, request.session_id)


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
