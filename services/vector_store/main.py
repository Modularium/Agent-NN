"""FastAPI entrypoint for the Vector Store service."""

from fastapi import FastAPI

from core.logging_utils import LoggingMiddleware, exception_handler, init_logging
from core.metrics_utils import MetricsMiddleware, metrics_router
from core.auth_utils import AuthMiddleware

from ..health_router import health_router
from .config import settings
from .routes import router as vector_router

logger = init_logging("vector_store")
app = FastAPI(title="Vector Store Service")
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(AuthMiddleware, logger=logger)
app.add_middleware(MetricsMiddleware, service="vector_store")
app.add_exception_handler(Exception, exception_handler(logger))
app.include_router(metrics_router())
app.include_router(health_router)
app.include_router(vector_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
