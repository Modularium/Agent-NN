from __future__ import annotations

import os
from typing import Dict

from .agent_profile import AgentIdentity
from .reputation import aggregate_score
from .trust_network import load_recommendations

MIN_REP = float(os.getenv("TRUST_CIRCLE_REP", "0.5"))
MIN_RECS = int(os.getenv("TRUST_CIRCLE_MIN", "2"))


def is_trusted_for(agent_id: str, role: str) -> bool:
    """Return True if ``agent_id`` is trusted for ``role``."""
    profile = AgentIdentity.load(agent_id)
    rep = profile.reputation_score or aggregate_score(agent_id)
    recs = [r for r in load_recommendations(agent_id) if r.role == role]
    if len(recs) < MIN_RECS:
        return False
    avg_conf = sum(r.confidence for r in recs) / len(recs)
    return rep >= MIN_REP and avg_conf >= 0.6
