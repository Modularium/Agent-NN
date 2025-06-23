"""FastAPI entrypoint for the Session Manager service."""

from fastapi import FastAPI

from core.run_service import run_service

from core.logging_utils import LoggingMiddleware, exception_handler, init_logging
from core.metrics_utils import MetricsMiddleware, metrics_router
from core.auth_utils import AuthMiddleware

from ..health_router import health_router
from .config import settings
from .routes import router as session_router

logger = init_logging("session_manager")
app = FastAPI(title="Session Manager Service")
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(AuthMiddleware, logger=logger)
app.add_middleware(MetricsMiddleware, service="session_manager")
app.add_exception_handler(Exception, exception_handler(logger))
app.include_router(metrics_router())
app.include_router(health_router)
app.include_router(session_router)

if __name__ == "__main__":
    run_service(app, host=settings.host, port=settings.port)
