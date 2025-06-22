from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .audit_log import AuditEntry, AuditLog
from .agent_profile import AgentIdentity
from .training import load_training_path
from .teams import AgentTeam

KNOWLEDGE_DIR = Path(os.getenv("TEAM_KNOWLEDGE_DIR", "team_knowledge"))


def _knowledge_file(team_id: str, skill: str) -> Path:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return KNOWLEDGE_DIR / f"{team_id}_{skill}.json"


def broadcast_insight(agent_id: str, skill: str, context: Dict) -> None:
    """Share ``context`` with all members of the agent's team."""

    profile = AgentIdentity.load(agent_id)
    if not profile.team_id:
        return
    team = AgentTeam.load(profile.team_id)
    path = _knowledge_file(team.id, skill)
    data: List[Dict] = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    data.append({"agent": agent_id, "context": context})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log = AuditLog()
    log.write(
        AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            actor=agent_id,
            action="shared_insight",
            context_id=team.id,
            detail={"skill": skill},
        )
    )
    _check_cooperative_success(team, skill)


def share_training_material(team_id: str, skill: str) -> List[Dict]:
    """Return shared training data for ``skill`` within ``team_id``."""

    path = _knowledge_file(team_id, skill)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _check_cooperative_success(team: AgentTeam, skill: str) -> None:
    tp = load_training_path(skill)
    if not tp or tp.team_mode != "cooperative":
        return
    if all(
        AgentIdentity.load(m).training_progress.get(skill) == "complete"
        for m in team.members
    ):
        log = AuditLog()
        log.write(
            AuditEntry(
                timestamp=datetime.utcnow().isoformat(),
                actor="system",
                action="cooperative_training_success",
                context_id=team.id,
                detail={"skill": skill},
            )
        )

