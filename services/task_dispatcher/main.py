"""FastAPI entrypoint for the Task Dispatcher service."""

from fastapi import FastAPI

from core.logging_utils import LoggingMiddleware, exception_handler, init_logging
from core.metrics_utils import MetricsMiddleware, metrics_router

from ..health_router import health_router
from .config import settings
from .routes import router as task_router

logger = init_logging("task_dispatcher")
app = FastAPI(title="Task Dispatcher Service")
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(MetricsMiddleware, service="task_dispatcher")
app.add_exception_handler(Exception, exception_handler(logger))
app.include_router(metrics_router())
app.include_router(health_router)
app.include_router(task_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
