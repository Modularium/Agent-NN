"""FastAPI entrypoint for the Vector Store service."""

from fastapi import FastAPI

from ..health_router import health_router
from .config import settings
from .routes import router as vector_router

app = FastAPI(title="Vector Store Service")
app.include_router(health_router)
app.include_router(vector_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
