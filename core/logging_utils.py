import logging
import os
import sys
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Callable

from fastapi import FastAPI

import structlog
from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings


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

    log_dir = os.getenv("LOG_DIR", settings.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    file_path = os.path.join(log_dir, f"{service}.log")
    handlers.append(RotatingFileHandler(file_path, maxBytes=1_048_576, backupCount=3))
    logging.basicConfig(level=level, handlers=handlers)
    return structlog.get_logger().bind(service=service)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests with a request id."""

    def __init__(self, app, logger: structlog.BoundLogger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        log = self.logger.bind(
            request_id=request_id,
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
        level = (
            logging.INFO
            if response.status_code < 400
            else logging.WARNING if response.status_code < 500 else logging.ERROR
        )
        log.log(
            level,
            event="request",
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration=duration,
        )
        return response


# backwards compatibility alias
LoggingMiddleware = RequestLoggingMiddleware


def exception_handler(logger: structlog.BoundLogger):
    """Return a FastAPI exception handler using the given logger."""

    async def handle(request: Request, exc: Exception):
        status = exc.status_code if isinstance(exc, HTTPException) else 500
        level = (
            logging.ERROR
            if status >= 500
            else logging.WARNING if status >= 400 else logging.INFO
        )
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
                "status": "error",
                "error": exc.__class__.__name__,
                "detail": str(exc),
                "timestamp": datetime.utcnow().isoformat(),
                "context": {"path": request.url.path},
            },
        )

    return handle


def register_shutdown_task(app: FastAPI, func: Callable[[], None]) -> None:
    """Run ``func`` when the FastAPI app shuts down."""

    @app.on_event("shutdown")
    async def _run() -> None:  # pragma: no cover - best effort
        try:
            func()
        except Exception:
            logging.getLogger(__name__).exception("shutdown_task_failed")
