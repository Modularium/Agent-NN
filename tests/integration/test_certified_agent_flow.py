import json
import os
import subprocess
import sys
from pathlib import Path

from core.agent_profile import AgentIdentity


def test_certified_agent_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("skills", exist_ok=True)
    with open("skills/demo.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "id": "demo",
                "title": "Demo",
                "required_for_roles": ["writer"],
                "expires_at": None,
            },
            fh,
        )

    os.environ["AGENT_PROFILE_DIR"] = str(tmp_path / "profiles")
    profile = AgentIdentity(
        name="demo",
        role="writer",
        traits={},
        skills=[],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[],
    )
    profile.save()

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "agentnn"),
            "agent",
            "certify",
            "demo",
            "--skill",
            "demo",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    profile = AgentIdentity.load("demo")
    assert profile.certified_skills and profile.certified_skills[0]["id"] == "demo"
