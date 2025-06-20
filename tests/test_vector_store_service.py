import sys
from types import ModuleType

embed_mod = ModuleType("embed")

class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, text):
        return [len(text)]

embed_mod.HuggingFaceEmbeddings = FakeEmbeddings
sys.modules.setdefault("langchain_community.embeddings", embed_mod)
from mcp.vector_store.service import VectorStoreService


def test_add_and_query(monkeypatch):
    service = VectorStoreService()
    service.add_document({"id": "a", "text": "foo"})
    service.add_document({"id": "b", "text": "bar"})
    res = service.query("foo")
    ids = {r["id"] for r in res}
    assert {"a", "b"}.issubset(ids)


def test_empty_query(monkeypatch):
    service = VectorStoreService()
    assert service.query("") == []
