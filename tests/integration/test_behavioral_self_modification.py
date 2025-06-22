from core.feedback_loop import FeedbackLoopEntry, record_feedback, load_feedback
from core.self_reflection import reflect_and_adapt
from core.agent_profile import AgentIdentity


def test_behavioral_modification(tmp_path):
    profile = AgentIdentity(
        name="agent",
        role="",
        traits={"assertiveness": 1.0},
        skills=[],
        memory_index=None,
        created_at="now",
    )
    base = tmp_path / "fb"
    for _ in range(3):
        record_feedback(
            FeedbackLoopEntry(
                agent_id="agent",
                event_type="task_failed",
                data={},
                created_at="x",
            ),
            base_dir=str(base),
        )
    feedback = load_feedback("agent", base_dir=str(base))
    result = reflect_and_adapt(profile, feedback)
    profile.traits.update(result.get("traits", {}))
    assert profile.traits["assertiveness"] < 1.0
