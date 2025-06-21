"""FastAPI entrypoint for the Session Manager service."""

from fastapi import FastAPI

from ..health_router import health_router
from .config import settings
from .routes import router as session_router

app = FastAPI(title="Session Manager Service")
app.include_router(health_router)
app.include_router(session_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
