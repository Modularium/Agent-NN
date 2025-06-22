import json
import os
import subprocess
import sys
from pathlib import Path

from core.agent_profile import AgentIdentity


def test_peer_rating_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.environ["RATING_DIR"] = str(tmp_path / "ratings")
    os.environ["AGENT_PROFILE_DIR"] = str(tmp_path / "profiles")

    for name in ["mentor", "analyst"]:
        AgentIdentity(
            name=name,
            role="",
            traits={},
            skills=[],
            memory_index=None,
            created_at="now",
        ).save()

    script = Path(__file__).resolve().parents[2] / "scripts" / "agentnn"
    subprocess.run(
        [sys.executable, str(script), "rate", "mentor", "analyst", "--score", "0.9"],
        check=True,
    )
    result = subprocess.run(
        [sys.executable, str(script), "agent", "rep", "analyst"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["reputation"] > 0
    lb = subprocess.run(
        [sys.executable, str(script), "rep", "leaderboard"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "analyst" in lb.stdout
