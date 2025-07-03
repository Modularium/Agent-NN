"""Lightweight MCP compatible server exposing context and execution endpoints."""

from __future__ import annotations

import os
from fastapi import APIRouter, FastAPI

from api_gateway.connectors import ServiceConnector
from ..storage import context_store
from core.model_context import ModelContext
from core.run_service import run_service

DISPATCHER_URL = os.getenv("DISPATCHER_URL", "http://task_dispatcher:8000")
SESSION_MANAGER_URL = os.getenv("SESSION_MANAGER_URL", "http://session_manager:8000")
REGISTRY_URL = os.getenv("AGENT_REGISTRY_URL", "http://agent_registry:8001")
TOOLS_URL = os.getenv("PLUGIN_AGENT_URL", "http://plugin_agent_service:8110")


def create_app() -> FastAPI:
    """Return the FastAPI application."""
    router = APIRouter(prefix="/v1/mcp")
    dispatcher = ServiceConnector(DISPATCHER_URL)
    sessions = ServiceConnector(SESSION_MANAGER_URL)
    registry = ServiceConnector(REGISTRY_URL)
    tools = ServiceConnector(TOOLS_URL)

    @router.get("/ping")
    async def ping() -> dict:
        return {"status": "ok"}

    @router.post("/execute", response_model=ModelContext)
    async def execute(ctx: ModelContext) -> ModelContext:
        return await dispatcher.post("/dispatch", ctx.model_dump())

    @router.post("/task/execute", response_model=ModelContext)
    async def execute_task(ctx: ModelContext) -> ModelContext:
        return await dispatcher.post("/dispatch", ctx.model_dump())

    @router.post("/context")
    async def update_context(ctx: ModelContext) -> dict:
        await sessions.post("/update_context", ctx.model_dump())
        return {"status": "ok"}

    @router.post("/context/save")
    async def save_context_route(ctx: ModelContext) -> dict:
        await sessions.post("/update_context", ctx.model_dump())
        context_store.save_context(ctx.session_id or "default", ctx.model_dump())
        return {"status": "ok"}

    @router.get("/context/load/{sid}")
    async def load_context_route(sid: str) -> dict:
        return {"context": context_store.load_context(sid)}

    @router.get("/context/history")
    async def list_contexts_route() -> dict:
        return {"sessions": context_store.list_contexts()}

    @router.get("/context/{sid}")
    async def get_context(sid: str) -> dict:
        return await sessions.get(f"/context/{sid}")

    @router.get("/context/get/{sid}")
    async def get_context_alt(sid: str) -> dict:
        return await sessions.get(f"/context/{sid}")

    @router.post("/agent/create")
    async def create_agent(agent: dict) -> dict:
        return await registry.post("/register", agent)

    @router.post("/tool/use")
    async def use_tool(payload: dict) -> dict:
        return await tools.post("/execute_tool", payload)

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
