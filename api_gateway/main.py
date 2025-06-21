import os
import json
import urllib.request
from fastapi import FastAPI, Request, HTTPException, Depends
from utils.api_utils import api_route
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator

from core.logging_utils import LoggingMiddleware, exception_handler, init_logging
from core.metrics_utils import MetricsMiddleware, metrics_router
from jose import JWTError, jwt


API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"
API_GATEWAY_KEY = os.getenv("API_GATEWAY_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
API_KEY_SCOPES = os.getenv("API_KEY_SCOPES", "*").split(",")
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://llm_gateway:8004")
DISPATCHER_URL = os.getenv("DISPATCHER_URL", "http://task_dispatcher:8000")
SESSION_MANAGER_URL = os.getenv("SESSION_MANAGER_URL", "http://session_manager:8000")
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")

logger = init_logging("api_gateway")
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
app = FastAPI(title="API Gateway")
app.add_middleware(LoggingMiddleware, logger=logger)
app.add_middleware(MetricsMiddleware, service="api_gateway")
app.add_exception_handler(Exception, exception_handler(logger))
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOW_ORIGINS.split(","), allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app)
app.include_router(metrics_router())


def _decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:  # pragma: no cover - invalid tokens
        return None


def check_scope(request: Request, scope: str) -> None:
    """Verify that the caller is authorized for the given scope."""
    if not API_AUTH_ENABLED:
        return
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key == API_GATEWAY_KEY:
        if scope in API_KEY_SCOPES or "*" in API_KEY_SCOPES:
            return
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        payload = _decode_token(token.split()[1])
        if payload:
            scopes = payload.get("scopes", [])
            if scope in scopes or "*" in scopes:
                return
    raise HTTPException(status_code=403, detail="Forbidden")


@api_route(version="dev")  # \U0001F6A7 experimental
@app.post("/llm/generate")
@limiter.limit(RATE_LIMIT)
async def llm_generate(request: Request) -> dict:
    check_scope(request, "llm:generate")
    payload = await request.body()
    url = f"{LLM_GATEWAY_URL}/generate"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return {"text": resp.read().decode()}


@api_route(version="dev")  # \U0001F6A7 experimental
@app.post("/chat")
@limiter.limit(RATE_LIMIT)
async def chat(request: Request) -> dict:
    check_scope(request, "chat:write")
    payload = await request.json()
    sid = payload.get("session_id")
    if not sid:
        req = urllib.request.Request(
            f"{SESSION_MANAGER_URL}/session",
            data=json.dumps({"data": {}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            sid = json.loads(resp.read().decode())["session_id"]

    data = {
        "task_type": "chat",
        "input": payload.get("message", ""),
        "session_id": sid,
    }
    req = urllib.request.Request(
        f"{DISPATCHER_URL}/task",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
    return {"session_id": sid, **result}


@api_route(version="dev")  # \U0001F6A7 experimental
@app.get("/chat/history/{sid}")
@limiter.limit(RATE_LIMIT)
async def chat_history(sid: str, request: Request) -> dict:
    check_scope(request, "chat:read")
    url = f"{SESSION_MANAGER_URL}/session/{sid}/history"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    return data


@api_route(version="dev")  # \U0001F6A7 experimental
@app.post("/chat/feedback")
@limiter.limit(RATE_LIMIT)
async def chat_feedback(request: Request) -> dict:
    check_scope(request, "feedback:write")
    payload = await request.json()
    sid = payload.get("session_id")
    url = f"{SESSION_MANAGER_URL}/session/{sid}/feedback"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())
