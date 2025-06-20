import os
import urllib.request
from fastapi import FastAPI, Request, HTTPException

API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"
API_GATEWAY_KEY = os.getenv("API_GATEWAY_KEY", "")
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL", "http://llm_gateway:8004")

app = FastAPI(title="API Gateway")


def check_key(request: Request) -> None:
    if API_AUTH_ENABLED and request.headers.get("X-API-Key") != API_GATEWAY_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/llm/generate")
async def llm_generate(request: Request) -> dict:
    check_key(request)
    payload = await request.body()
    url = f"{LLM_GATEWAY_URL}/generate"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return {"text": resp.read().decode()}
