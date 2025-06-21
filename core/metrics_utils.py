import time
from fastapi import APIRouter, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

TASKS_PROCESSED = Counter(
    "agentnn_tasks_processed_total", "Total processed tasks", ["service"]
)
ACTIVE_SESSIONS = Gauge(
    "agentnn_active_sessions", "Active sessions", ["service"]
)
TOKENS_IN = Counter("agentnn_tokens_in_total", "Tokens received", ["service"])
TOKENS_OUT = Counter("agentnn_tokens_out_total", "Tokens sent", ["service"])
RESPONSE_TIME = Histogram(
    "agentnn_response_seconds", "Response time in seconds", ["service", "path"]
)


def metrics_router() -> APIRouter:
    router = APIRouter()

    @router.get("/metrics")
    async def metrics() -> Response:
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return router


class MetricsMiddleware(BaseHTTPMiddleware):
    """Track request durations."""

    def __init__(self, app, service: str):
        super().__init__(app)
        self.service = service

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        RESPONSE_TIME.labels(self.service, request.url.path).observe(duration)
        return response
