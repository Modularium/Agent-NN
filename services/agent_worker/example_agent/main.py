"""FastAPI entrypoint for Example Agent Worker."""

from fastapi import FastAPI

from ...health_router import health_router
from .config import settings
from .routes import router as agent_router

app = FastAPI(title="Example Agent Worker")
app.include_router(health_router)
app.include_router(agent_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
