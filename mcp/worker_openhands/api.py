from fastapi import APIRouter
from pydantic import BaseModel

from .service import WorkerService

router = APIRouter()
service = WorkerService()


class TaskRequest(BaseModel):
    task: str


@router.post("/execute_task")
async def execute(request: TaskRequest) -> dict:
    result = service.execute_task(request.task)
    return {"result": result}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
