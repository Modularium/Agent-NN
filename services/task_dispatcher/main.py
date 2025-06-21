"""FastAPI entrypoint for the Task Dispatcher service."""

from fastapi import FastAPI

from ..health_router import health_router
from .config import settings
from .routes import router as task_router

app = FastAPI(title="Task Dispatcher Service")
app.include_router(health_router)
app.include_router(task_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
