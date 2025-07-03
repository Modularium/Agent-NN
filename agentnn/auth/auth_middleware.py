"""Simple header based authentication middleware."""

from __future__ import annotations

from typing import List

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate API key or bearer token from request headers."""

    def __init__(self, app, api_keys: str = "", bearer_token: str | None = None) -> None:
        super().__init__(app)
        self.keys: List[str] = [k.strip() for k in api_keys.split(",") if k.strip()]
        self.bearer = bearer_token

    async def dispatch(self, request: Request, call_next):
        authorized = False
        key = request.headers.get("X-API-Key")
        if key and (not self.keys or key in self.keys):
            authorized = True
        token = request.headers.get("Authorization")
        if token and token.startswith("Bearer ") and self.bearer:
            if token.split()[1] == self.bearer:
                authorized = True
        if not self.keys and not self.bearer:
            authorized = True
        if not authorized:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return await call_next(request)
