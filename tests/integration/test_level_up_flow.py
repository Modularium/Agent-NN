import json
import os
import subprocess
import sys
from pathlib import Path

from core.agent_profile import AgentIdentity


def test_level_up_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("levels", exist_ok=True)
    with open("levels/basic.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "id": "basic",
                "title": "Basic",
                "trust_required": 0.5,
                "skills_required": ["demo"],
                "unlocks": {"roles": ["reviewer"]},
            },
            fh,
        )
    os.environ["AGENT_PROFILE_DIR"] = str(tmp_path / "profiles")
    os.environ["CONTRACT_DIR"] = str(tmp_path / "contracts")
    profile = AgentIdentity(
        name="demo",
        role="writer",
        traits={},
        skills=[],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[{"id": "demo", "granted_at": "now", "expires_at": None}],
    )
    profile.save()
    history = [
        {
            "agent_id": "demo",
            "success": True,
            "feedback_score": 1.0,
            "metrics": {"tokens_used": 1},
            "expected_tokens": 1,
            "error": None,
        }
        for _ in range(5)
    ]
    from core.governance import AgentContract

    AgentContract(
        agent="demo",
        allowed_roles=["writer"],
        max_tokens=0,
        trust_level_required=0.0,
        constraints={"task_history": history},
    ).save()

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "agentnn"),
            "agent",
            "promote",
            "demo",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    profile = AgentIdentity.load("demo")
    assert profile.current_level == "basic"
