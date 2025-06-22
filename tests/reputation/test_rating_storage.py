from datetime import datetime
import importlib


def test_rating_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("RATING_DIR", str(tmp_path))
    mod = importlib.import_module("core.reputation")
    importlib.reload(mod)
    rating = mod.AgentRating(
        from_agent="a1",
        to_agent="a2",
        mission_id=None,
        rating=0.8,
        feedback="good",
        context_tags=["clarity"],
        created_at=datetime.utcnow().isoformat(),
    )
    mod.save_rating(rating)
    items = mod.load_ratings("a2")
    assert items and items[0].from_agent == "a1"
