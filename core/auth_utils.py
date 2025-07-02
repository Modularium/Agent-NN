import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
import httpx


def _load_tokens() -> set[str]:
    tokens = os.getenv("API_TOKENS", "")
    return {t.strip() for t in tokens.split(",") if t.strip()}


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple bearer token authentication middleware."""

    def __init__(self, app, logger: structlog.BoundLogger):
        super().__init__(app)
        self.logger = logger
        self.enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"
        self.tokens = _load_tokens()
        self.user_service_url = os.getenv("USER_SERVICE_URL")
        self.exempt_paths = {"/health", "/metrics", "/status"}

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or request.url.path in self.exempt_paths:
            return await call_next(request)
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            self.logger.warning(
                "auth_missing",
                event="auth_missing",
                path=request.url.path,
                ip=getattr(request.client, "host", ""),
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(status_code=401, detail="Unauthorized")
        token = auth.split()[1]
        valid = token in self.tokens
        if not valid and self.user_service_url:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{self.user_service_url.rstrip('/')}/validate",
                        json={"token": token},
                        timeout=5,
                    )
                    valid = resp.status_code == 200
            except Exception:  # pragma: no cover - network failures
                valid = False
        if not valid:
            self.logger.warning(
                "auth_invalid",
                event="auth_invalid",
                path=request.url.path,
                ip=getattr(request.client, "host", ""),
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(status_code=403, detail="Forbidden")
        return await call_next(request)
