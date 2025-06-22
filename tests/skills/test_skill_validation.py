from datetime import datetime, timedelta

from core.agent_profile import AgentIdentity
from core.skill_matcher import match_agent_to_task


def test_skill_match_and_expiry():
    profile = AgentIdentity(
        name="a",
        role="writer",
        traits={},
        skills=[],
        memory_index=None,
        created_at="2024-01-01T00:00:00Z",
        certified_skills=[
            {
                "id": "demo",
                "granted_at": datetime.utcnow().isoformat() + "Z",
                "expires_at": (datetime.utcnow() + timedelta(days=1)).isoformat() + "Z",
            }
        ],
    )
    assert match_agent_to_task(profile, ["demo"]) is True
    profile.certified_skills[0]["expires_at"] = (
        datetime.utcnow() - timedelta(days=1)
    ).isoformat() + "Z"
    assert match_agent_to_task(profile, ["demo"]) is False
