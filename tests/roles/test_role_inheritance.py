from core.access_control import is_authorized
from core.governance import AgentContract


def test_role_inheritance(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_DIR", str(tmp_path))
    AgentContract(
        agent="demo",
        allowed_roles=["critic"],
        temp_roles=None,
        max_tokens=0,
        trust_level_required=0.0,
        constraints={},
    ).save()

    assert is_authorized("demo", "reviewer", "view_context", "demo")
    assert not is_authorized("demo", "writer", "submit_task", "demo")
