from fastapi.testclient import TestClient
from services.vector_store.routes import router, service
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_search_route(monkeypatch):
    monkeypatch.setattr(service, "search", lambda q, c, top_k=3: {"matches": [{"id": "1"}], "model": "dummy"})
    resp = client.post("/vector_search", json={"query": "hi", "collection": "c", "top_k": 1})
    assert resp.status_code == 200
    assert resp.json()["matches"][0]["id"] == "1"


def test_embed_route(monkeypatch):
    monkeypatch.setattr(service, "embed", lambda text: {"embedding": [1.0], "model": "dummy"})
    resp = client.post("/embed", json={"text": "hi"})
    assert resp.status_code == 200
    assert resp.json()["embedding"] == [1.0]
