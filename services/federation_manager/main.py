"""FastAPI entrypoint for the Federation Manager service."""

from fastapi import FastAPI

from core.run_service import run_service
from core.logging_utils import LoggingMiddleware, exception_handler, init_logging
from core.metrics_utils import MetricsMiddleware, metrics_router
from core.auth_utils import AuthMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..health_router import health_router
from .config import settings
from .routes import router as fed_router

logger = init_logging("federation_manager")
app = FastAPI(title="Federation Manager Service")
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(AuthMiddleware, logger=logger)
app.add_middleware(MetricsMiddleware, service="federation_manager")
app.add_exception_handler(Exception, exception_handler(logger))
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(metrics_router())
app.include_router(health_router)
app.include_router(fed_router)

if __name__ == "__main__":
    run_service(app, host=settings.host, port=settings.port)
