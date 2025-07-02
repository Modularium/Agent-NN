from fastapi.testclient import TestClient
from fastapi import FastAPI

from services.user_manager.routes import router, service


def create_client(tmp_path) -> TestClient:
    service.path = tmp_path / "users.json"
    service.users.clear()
    service.tokens.clear()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_register_login_validate(tmp_path):
    client = create_client(tmp_path)
    resp = client.post("/register", json={"username": "alice", "password": "pw"})
    assert resp.status_code == 200

    resp = client.post("/login", json={"username": "alice", "password": "pw"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert token

    resp = client.post("/validate", json={"token": token})
    assert resp.status_code == 200
