from datetime import datetime, timedelta

from core.agent_profile import AgentIdentity
from core.delegation import grant_delegation, has_valid_delegation


def test_expired_delegation(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE_DIR", str(tmp_path / "profiles"))
    monkeypatch.setenv("DELEGATION_DIR", str(tmp_path / "delegations"))
    AgentIdentity(name="d1", role="", traits={}, skills=[], memory_index=None, created_at="now").save()
    AgentIdentity(name="d2", role="", traits={}, skills=[], memory_index=None, created_at="now").save()

    expired = (datetime.utcnow() - timedelta(days=1)).isoformat()
    grant_delegation("d1", "d2", "writer", "task", expires_at=expired)
    assert has_valid_delegation("d2", "writer") is None
