from services.agent_worker.demo_agents.writer_agent import WriterAgent
from core.agent_profile import AgentIdentity


def test_profile_loading(monkeypatch):
    called = {}

    def fake_load(name: str) -> AgentIdentity:
        called["name"] = name
        return AgentIdentity(
            name=name,
            role="writer",
            traits={},
            skills=[],
            memory_index=None,
            created_at="now",
        )

    monkeypatch.setattr(AgentIdentity, "load", classmethod(lambda cls, n: fake_load(n)))
    WriterAgent(llm_url="http://llm")
    assert called["name"] == "writer_agent"
