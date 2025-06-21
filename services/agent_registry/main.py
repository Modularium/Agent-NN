"""FastAPI entrypoint for the Agent Registry service."""

from fastapi import FastAPI

from ..health_router import health_router
from .config import settings
from .routes import router as registry_router

app = FastAPI(title="Agent Registry Service")
app.include_router(health_router)
app.include_router(registry_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
