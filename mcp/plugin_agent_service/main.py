import uvicorn
from fastapi import FastAPI

from .api import router

app = FastAPI(title="Plugin Agent Service")
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)
