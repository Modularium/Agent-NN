from core.feedback_loop import FeedbackLoopEntry, record_feedback, load_feedback


def test_record_and_load(tmp_path):
    entry = FeedbackLoopEntry(
        agent_id="tester",
        event_type="task_failed",
        data={"reason": "error"},
        created_at="2024-01-01T00:00:00",
    )
    base = tmp_path / "fb"
    record_feedback(entry, base_dir=str(base))
    entries = load_feedback("tester", base_dir=str(base))
    assert len(entries) == 1
    assert entries[0].event_type == "task_failed"
