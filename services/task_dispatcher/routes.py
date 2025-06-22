"""API routes for the Task Dispatcher service."""

import os

from fastapi import APIRouter
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.model_context import ModelContext
from utils.api_utils import api_route

from .schemas import TaskRequest
from .service import TaskDispatcherService

router = APIRouter()
service = TaskDispatcherService()
RATE_LIMIT_TASK = os.getenv("RATE_LIMIT_TASK", "10/minute")
RATE_LIMITS_ENABLED = os.getenv("RATE_LIMITS_ENABLED", "true").lower() == "true"
limiter = Limiter(key_func=get_remote_address)
limit_task = limiter.limit(RATE_LIMIT_TASK) if RATE_LIMITS_ENABLED else (lambda f: f)


@api_route(version="v1.0.0")
@router.post("/task", response_model=ModelContext)
@limit_task
async def create_task(task: TaskRequest) -> ModelContext:
    """Accept a TaskContext and queue it for processing."""
    return service.dispatch_task(
        task,
        session_id=task.session_id,
        mode=task.mode,
        task_value=task.task_value,
        max_tokens=task.max_tokens,
        priority=task.priority,
        deadline=task.deadline,
    )


@api_route(version="v1.0.0")
@router.post("/queue/promote/{task_id}")
async def promote(task_id: str) -> dict:
    """Promote a task within the queue."""
    service.queue.promote_high_priority()
    return {"promoted": task_id}


@api_route(version="v1.0.0")
@router.get("/queue/status")
async def queue_status() -> list[dict]:
    """Return queued tasks."""
    return [ctx.model_dump() for ctx in service.queue._queue]
