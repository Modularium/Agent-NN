from services.agent_worker.demo_agents.critic_agent import CriticAgent


def test_vote_returns_score_and_feedback():
    agent = CriticAgent()
    result = agent.vote("some long text for testing", "quality")
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["feedback"], str)
