import os
import urllib.request
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from jose import JWTError, jwt


API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"
API_GATEWAY_KEY = os.getenv("API_GATEWAY_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://llm_gateway:8004")
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
app = FastAPI(title="API Gateway")
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=CORS_ALLOW_ORIGINS.split(","), allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app)


def check_key(request: Request) -> None:
    if not API_AUTH_ENABLED:
        return
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key == API_GATEWAY_KEY:
        return
    token = request.headers.get("Authorization")
    if token and token.startswith("Bearer "):
        try:
            jwt.decode(token.split()[1], JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return
        except JWTError:
            pass
    raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/llm/generate")
@limiter.limit(RATE_LIMIT)
async def llm_generate(request: Request) -> dict:
    check_key(request)
    payload = await request.body()
    url = f"{LLM_GATEWAY_URL}/generate"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return {"text": resp.read().decode()}
