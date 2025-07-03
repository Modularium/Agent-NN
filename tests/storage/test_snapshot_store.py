import importlib.util
import pathlib


def test_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_BACKEND", "sqlite")
    monkeypatch.setenv("CONTEXT_DB_PATH", str(tmp_path / "ctx.db"))
    monkeypatch.setenv("SNAPSHOT_PATH", str(tmp_path))
    ctx_spec = importlib.util.spec_from_file_location(
        "context_store", pathlib.Path("agentnn/storage/context_store.py")
    )
    ctx = importlib.util.module_from_spec(ctx_spec)
    ctx_spec.loader.exec_module(ctx)  # type: ignore
    snap_spec = importlib.util.spec_from_file_location(
        "snapshot_store", pathlib.Path("agentnn/storage/snapshot_store.py")
    )
    snap = importlib.util.module_from_spec(snap_spec)
    snap_spec.loader.exec_module(snap)  # type: ignore

    ctx.save_context("s1", {"a": 1})
    snap_id = snap.save_snapshot("s1")
    new_session = snap.restore_snapshot(snap_id)
    items = ctx.load_context(new_session)
    assert items[0]["a"] == 1
