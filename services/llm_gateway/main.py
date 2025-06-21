"""FastAPI entrypoint for the LLM Gateway service."""

from fastapi import FastAPI

from ..health_router import health_router
from .config import settings
from .routes import router as llm_router

app = FastAPI(title="LLM Gateway Service")
app.include_router(health_router)
app.include_router(llm_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
