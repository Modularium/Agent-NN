from core.feedback_loop import FeedbackLoopEntry
from core.self_reflection import reflect_and_adapt
from core.agent_profile import AgentIdentity


def test_reflection_suggests_trait_change():
    profile = AgentIdentity(
        name="agent",
        role="",
        traits={"assertiveness": 1.0},
        skills=[],
        memory_index=None,
        created_at="now",
    )
    log = [
        FeedbackLoopEntry(
            agent_id="agent", event_type="task_failed", data={}, created_at="x"
        )
        for _ in range(3)
    ]
    result = reflect_and_adapt(profile, log)
    assert result["traits"].get("assertiveness") == 0.9
