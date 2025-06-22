import json
import os
import subprocess
import sys
from pathlib import Path

from core.agent_profile import AgentIdentity


def test_trust_circle_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.environ["RECOMMEND_DIR"] = str(tmp_path / "recs")
    os.environ["AGENT_PROFILE_DIR"] = str(tmp_path / "profiles")

    for name in ["mentor", "peer"]:
        AgentIdentity(
            name=name,
            role="",
            traits={},
            skills=[],
            memory_index=None,
            created_at="now",
        ).save()
    AgentIdentity(
        name="analyst",
        role="reviewer",
        traits={},
        skills=[],
        memory_index=None,
        created_at="now",
    ).save()

    script = Path(__file__).resolve().parents[2] / "scripts" / "agentnn"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "trust",
            "endorse",
            "mentor",
            "analyst",
            "--role",
            "reviewer",
            "--confidence",
            "0.8",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "trust",
            "endorse",
            "peer",
            "analyst",
            "--role",
            "reviewer",
            "--confidence",
            "0.9",
        ],
        check=True,
    )

    result = subprocess.run(
        [sys.executable, str(script), "trust", "circle", "analyst"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data.get("reviewer") is True
