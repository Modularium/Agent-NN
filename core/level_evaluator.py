from __future__ import annotations

from datetime import datetime
from typing import Optional

from .agent_profile import AgentIdentity
from .audit_log import AuditEntry, AuditLog
from .governance import AgentContract
from .levels import load_levels
from .rewards import grant_rewards
from .trust_evaluator import calculate_trust


def _agent_skills(agent: AgentIdentity) -> set[str]:
    return {c.get("id") for c in agent.certified_skills}


def check_level_up(agent: AgentIdentity) -> Optional[str]:
    """Check if ``agent`` qualifies for the next level.

    Returns the new level id or ``None``.
    """

    contract = AgentContract.load(agent.name)
    history = contract.constraints.get("task_history", [])
    trust = calculate_trust(agent.name, history)
    levels = load_levels()
    current = agent.current_level
    current_index = -1
    for idx, lvl in enumerate(levels):
        if lvl.id == current:
            current_index = idx
            break

    for level in levels[current_index + 1 :]:
        if trust < level.trust_required:
            continue
        if not set(level.skills_required).issubset(_agent_skills(agent)):
            continue

        agent.current_level = level.id
        agent.level_progress[level.id] = 1.0
        agent.save()

        for role in level.unlocks.get("roles", []):
            if role not in contract.allowed_roles:
                contract.allowed_roles.append(role)
        contract.save()

        grant_rewards(agent.name, level.unlocks.get("rewards", {}))

        log = AuditLog()
        log.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="system",
                action="level_up",
                context_id=agent.name,
                detail={"level": level.id},
            )
        )
        return level.id
    return None
