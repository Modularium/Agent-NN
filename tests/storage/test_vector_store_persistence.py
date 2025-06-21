import tempfile

import numpy as np
from services.vector_store.service import VectorStoreService


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class DummyClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def post(self, url, json, timeout=10):
        return DummyResp({"embedding": [float(len(json["text"]))], "provider": "dummy"})


def test_vector_store_persistence(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("VECTOR_DB_BACKEND", "chromadb")
        monkeypatch.setenv("VECTOR_DB_DIR", tmp)
        service = VectorStoreService(llm_url="http://llm")
        doc_id = service.add_document("foo", "test")
        service2 = VectorStoreService(llm_url="http://llm")
        res = service2.search("foo", "test", top_k=1)
        assert res["matches"][0]["id"] == doc_id
