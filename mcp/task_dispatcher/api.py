from fastapi import APIRouter
from pydantic import BaseModel

from .service import TaskDispatcherService

router = APIRouter()
service = TaskDispatcherService()


class TaskRequest(BaseModel):
    task: str


@router.post("/task")
async def create_task(request: TaskRequest):
    """Accept a new task and dispatch it."""
    return service.dispatch_task(request.task)


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
