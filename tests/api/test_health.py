from fastapi.testclient import TestClient
from fastapi import FastAPI
from services.health_router import health_router


def test_health_and_status():
    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
