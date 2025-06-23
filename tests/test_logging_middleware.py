import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
from core.logging_utils import init_logging, RequestLoggingMiddleware


def test_request_logging_middleware(capsys):
    os.environ["LOG_FORMAT"] = "json"
    logger = init_logging("test_service")
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, logger=logger)

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/ping")
    assert response.status_code == 200
    captured = capsys.readouterr().out
    assert "test_service" in captured
    assert "request" in captured
