from core.agent_profile import AgentIdentity
from core.delegation import grant_delegation, has_valid_delegation


def test_grant_and_use(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("DELEGATION_DIR", str(tmp_path / "delegations"))
    AgentIdentity(name="alice", role="coordinator", traits={}, skills=[], memory_index=None, created_at="now").save()
    AgentIdentity(name="bob", role="reviewer", traits={}, skills=[], memory_index=None, created_at="now").save()

    grant_delegation("alice", "bob", "reviewer", "task")
    grant = has_valid_delegation("bob", "reviewer")
    assert grant is not None and grant.delegator == "alice"

    alice = AgentIdentity.load("alice")
    bob = AgentIdentity.load("bob")
    assert bob.delegated_by == ["alice"]
    assert alice.active_delegations
