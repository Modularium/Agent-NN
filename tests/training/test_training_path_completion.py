from core.training import TrainingPath, load_training_path, save_training_path


def test_training_path_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tp = TrainingPath(
        id="demo",
        target_skill="review",
        prerequisites=[],
        method="prompt",
        evaluation_prompt="Test",
        certifier_agent="mentor",
        mentor_required=False,
        min_trust=0.5,
    )
    save_training_path(tp)
    loaded = load_training_path("demo")
    assert loaded == tp
