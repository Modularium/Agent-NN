"""Lightweight MCP compatible server exposing context and execution endpoints."""

from __future__ import annotations

import os
from fastapi import APIRouter, FastAPI

from api_gateway.connectors import ServiceConnector
from core.model_context import ModelContext
from core.run_service import run_service

DISPATCHER_URL = os.getenv("DISPATCHER_URL", "http://task_dispatcher:8000")
SESSION_MANAGER_URL = os.getenv("SESSION_MANAGER_URL", "http://session_manager:8000")


def create_app() -> FastAPI:
    """Return the FastAPI application."""
    router = APIRouter(prefix="/v1/mcp")
    dispatcher = ServiceConnector(DISPATCHER_URL)
    sessions = ServiceConnector(SESSION_MANAGER_URL)

    @router.get("/ping")
    async def ping() -> dict:
        return {"status": "ok"}

    @router.post("/execute", response_model=ModelContext)
    async def execute(ctx: ModelContext) -> ModelContext:
        return await dispatcher.post("/dispatch", ctx.model_dump())

    @router.post("/context")
    async def update_context(ctx: ModelContext) -> dict:
        await sessions.post("/update_context", ctx.model_dump())
        return {"status": "ok"}

    @router.get("/context/{sid}")
    async def get_context(sid: str) -> dict:
        return await sessions.get(f"/context/{sid}")

    app = FastAPI(title="Agent-NN MCP Server")
    app.include_router(router)
    return app


def main() -> None:  # pragma: no cover - entrypoint
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8090"))
    run_service(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover - script
    main()
