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
        if url.endswith("/embed"):
            return DummyResp({"embedding": [float(len(json["text"]))], "provider": "dummy"})
        raise AssertionError("unexpected url" + url)


def test_add_and_search(monkeypatch):
    monkeypatch.setattr("httpx.Client", lambda: DummyClient())
    service = VectorStoreService(llm_url="http://llm")
    a = service.add_document("foo", "test")
    b = service.add_document("bar", "test")
    res = service.search("foo", "test", top_k=2)
    ids = {m["id"] for m in res["matches"]}
    assert {a, b}.issubset(ids)
