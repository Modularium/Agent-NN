import importlib.util
import pathlib
import pytest


@pytest.mark.unit
def test_context_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_BACKEND", "sqlite")
    monkeypatch.setenv("CONTEXT_DB_PATH", str(tmp_path / "ctx.db"))
    spec = importlib.util.spec_from_file_location(
        "context_store", pathlib.Path("agentnn/storage/context_store.py")
    )
    store = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(store)  # type: ignore
    store.save_context("s1", {"a": 1})
    store.save_context("s1", {"b": 2})
    items = store.load_context("s1")
    assert items[0]["a"] == 1
    assert items[1]["b"] == 2
    assert "s1" in store.list_contexts()
