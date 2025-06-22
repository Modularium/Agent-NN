from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal

from .agent_profile import AgentIdentity


DELEGATION_DIR = Path(os.getenv("DELEGATION_DIR", "delegations"))


@dataclass
class DelegationGrant:
    delegator: str
    delegate: str
    role: str
    scope: Literal["task", "mission", "team", "permanent"]
    expires_at: Optional[str]
    reason: Optional[str]
    granted_at: str


def _path(delegator: str) -> Path:
    DELEGATION_DIR.mkdir(parents=True, exist_ok=True)
    return DELEGATION_DIR / f"{delegator}.jsonl"


def save_grant(grant: DelegationGrant) -> None:
    """Append ``grant`` to the delegator file."""
    path = _path(grant.delegator)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(grant)) + "\n")


def load_grants(delegator: str) -> List[DelegationGrant]:
    path = _path(delegator)
    grants: List[DelegationGrant] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                grants.append(DelegationGrant(**data))
    return grants


def revoke_grant(delegator: str, delegate: str, role: str) -> None:
    """Remove delegation to ``delegate`` for ``role``."""
    grants = [g for g in load_grants(delegator) if not (g.delegate == delegate and g.role == role)]
    path = _path(delegator)
    with open(path, "w", encoding="utf-8") as fh:
        for g in grants:
            fh.write(json.dumps(asdict(g)) + "\n")
    # update profiles
    delegator_profile = AgentIdentity.load(delegator)
    delegator_profile.active_delegations = [
        g for g in delegator_profile.active_delegations if not (g["delegate"] == delegate and g["role"] == role)
    ]
    delegator_profile.save()
    delegate_profile = AgentIdentity.load(delegate)
    if delegator in delegate_profile.delegated_by:
        if not any(g["delegator"] == delegator for g in delegator_profile.active_delegations):
            delegate_profile.delegated_by.remove(delegator)
            delegate_profile.save()


def grant_delegation(
    delegator: str,
    delegate: str,
    role: str,
    scope: str = "task",
    expires_at: Optional[str] = None,
    reason: Optional[str] = None,
) -> DelegationGrant:
    grant = DelegationGrant(
        delegator=delegator,
        delegate=delegate,
        role=role,
        scope=scope,
        expires_at=expires_at,
        reason=reason,
        granted_at=datetime.utcnow().isoformat(),
    )
    save_grant(grant)
    # update profiles
    delegator_profile = AgentIdentity.load(delegator)
    delegator_profile.active_delegations.append(asdict(grant))
    delegator_profile.save()
    delegate_profile = AgentIdentity.load(delegate)
    if delegator not in delegate_profile.delegated_by:
        delegate_profile.delegated_by.append(delegator)
        delegate_profile.save()
    return grant


def has_valid_delegation(
    agent: str,
    role: str,
    require_endorsement: bool = False,
) -> Optional[DelegationGrant]:
    """Return a valid delegation for ``agent`` and ``role`` if present."""
    now = datetime.utcnow().isoformat()
    for file in DELEGATION_DIR.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                data = json.loads(line)
                grant = DelegationGrant(**data)
                if grant.delegate != agent or grant.role != role:
                    continue
                if grant.expires_at and grant.expires_at < now:
                    continue
                if require_endorsement:
                    from .trust_circle import is_trusted_for

                    if not is_trusted_for(agent, role):
                        continue
                return grant
    return None
