from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

VOTE_DIR = Path(os.getenv("VOTE_DIR", "votes"))


@dataclass
class ProposalVote:
    """A single vote for a governance proposal."""

    proposal_id: str
    agent_id: str
    decision: str  # yes or no
    comment: Optional[str]
    created_at: str


def _path(pid: str) -> Path:
    VOTE_DIR.mkdir(parents=True, exist_ok=True)
    return VOTE_DIR / f"{pid}.jsonl"


def record_vote(vote: ProposalVote) -> None:
    """Persist ``vote`` in ``VOTE_DIR``."""
    path = _path(vote.proposal_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(vote)) + "\n")


def load_votes(pid: str) -> List[ProposalVote]:
    """Return all votes for ``pid``."""
    path = _path(pid)
    if not path.exists():
        return []
    votes: List[ProposalVote] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            votes.append(ProposalVote(**data))
    return votes
