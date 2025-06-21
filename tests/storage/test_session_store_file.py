import tempfile
from core.session_store import FileSessionStore


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileSessionStore(tmp)
        sid = store.start_session()
        store.update_context(sid, {"msg": 1})
        # reload
        store2 = FileSessionStore(tmp)
        assert store2.get_context(sid)[0]["msg"] == 1
