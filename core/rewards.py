from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from .audit_log import AuditEntry, AuditLog
from .governance import AgentContract


def grant_rewards(agent_id: str, rewards: Dict[str, Any]) -> None:
    """Apply ``rewards`` to ``agent_id`` and log the event."""

    if not rewards:
        return

    contract = AgentContract.load(agent_id)
    changed = False

    tokens = rewards.get("tokens")
    if tokens:
        contract.constraints["bonus_tokens"] = contract.constraints.get(
            "bonus_tokens", 0
        ) + int(tokens)
        changed = True

    for role in rewards.get("roles", []):
        if role not in contract.allowed_roles:
            contract.allowed_roles.append(role)
            changed = True

    if changed:
        contract.save()

    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor="system",
            action="reward_granted",
            context_id=agent_id,
            detail=rewards,
        )
    )
