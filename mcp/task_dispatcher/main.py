from fastapi import FastAPI

from core.run_service import run_service

from .api import router

app = FastAPI(title="Task Dispatcher Service")
app.include_router(router)

if __name__ == "__main__":
    run_service(app, host="0.0.0.0", port=8000)
