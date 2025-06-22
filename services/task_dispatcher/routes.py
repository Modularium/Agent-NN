"""API routes for the Task Dispatcher service."""

from fastapi import APIRouter
from utils.api_utils import api_route
import os
from slowapi.util import get_remote_address
from slowapi import Limiter

from .schemas import TaskRequest
from core.model_context import ModelContext
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
    )
