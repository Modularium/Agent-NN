import logging
import os
import sys
import time
from datetime import datetime
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import structlog


def init_logging(service: str) -> structlog.BoundLogger:
    """Configure structlog for the given service."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()
    level = getattr(logging, log_level, logging.INFO)

    processors = [
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
        structlog.processors.add_log_level,
    ]
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(stream=sys.stdout, level=level)
    return structlog.get_logger().bind(service=service)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests in a structured format."""

    def __init__(self, app, logger: structlog.BoundLogger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next: Callable):
        log = self.logger.bind(
            context_id=request.headers.get("x-context-id"),
            session_id=request.headers.get("x-session-id"),
            agent_id=request.headers.get("x-agent-id"),
        )
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - unexpected errors
            duration = time.perf_counter() - start
            log.error(
                "request_error",
                event="request_error",
                path=request.url.path,
                method=request.method,
                error=str(exc),
                duration=duration,
            )
            raise
        duration = time.perf_counter() - start
        level = "info" if response.status_code < 400 else "warning" if response.status_code < 500 else "error"
        log.log(
            level,
            "request",
            event="request",
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration=duration,
        )
        return response


def exception_handler(logger: structlog.BoundLogger):
    """Return a FastAPI exception handler using the given logger."""

    async def handle(request: Request, exc: Exception):
        status = exc.status_code if isinstance(exc, HTTPException) else 500
        level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
        logger.log(
            level,
            "exception",
            event="exception",
            path=request.url.path,
            error=str(exc),
            status_code=status,
        )
        return JSONResponse(
            status_code=status,
            content={
                "error": exc.__class__.__name__,
                "detail": str(exc),
                "timestamp": datetime.utcnow().isoformat(),
                "context": {"path": request.url.path},
            },
        )

    return handle
