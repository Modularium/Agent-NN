from fastapi.testclient import TestClient
from services.llm_gateway.routes import router, service
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_generate_route(monkeypatch):
    monkeypatch.setattr(service, "generate", lambda prompt, model_name=None, temperature=0.7: {"completion": "hi", "tokens_used": 1, "provider": "dummy"})
    resp = client.post("/generate", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["completion"] == "hi"


def test_embed_route(monkeypatch):
    monkeypatch.setattr(service, "embed", lambda text, model_name=None: {"embedding": [1.0], "provider": "dummy"})
    resp = client.post("/embed", json={"text": "hi"})
    assert resp.status_code == 200
    assert resp.json()["embedding"] == [1.0]
