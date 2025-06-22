from __future__ import annotations

from datetime import datetime
from typing import List

from .agent_profile import AgentIdentity


def match_agent_to_task(agent: AgentIdentity, required_skills: List[str]) -> bool:
    """Return ``True`` if agent holds valid certifications for all skills."""
    if not required_skills:
        return True
    now = datetime.utcnow()
    skills = {s.get("id"): s for s in agent.certified_skills}
    for skill_id in required_skills:
        cert = skills.get(skill_id)
        if not cert:
            return False
        expires = cert.get("expires_at")
        if expires and datetime.fromisoformat(expires) < now:
            return False
    return True
