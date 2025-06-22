from core.access_control import is_authorized
from core.governance import AgentContract


def test_is_authorized_role(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="demo",
        allowed_roles=["writer"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={},
    ).save()

    assert is_authorized("demo", "writer", "submit_task", "demo")
    assert not is_authorized("demo", "critic", "submit_task", "demo")
