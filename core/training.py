from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal

TRAINING_DIR = Path("training_paths")


@dataclass
class TrainingPath:
    """Definition of a skill training path."""

    id: str
    target_skill: str
    prerequisites: List[str]
    method: Literal["prompt", "task_simulation", "peer_review"]
    evaluation_prompt: str
    certifier_agent: str
    mentor_required: bool
    min_trust: float


def load_training_path(path_id: str) -> TrainingPath | None:
    """Load a ``TrainingPath`` from ``training_paths/{id}.json``."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAINING_DIR / f"{path_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return TrainingPath(**data)


def save_training_path(training: TrainingPath) -> None:
    """Persist ``training`` in the registry."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAINING_DIR / f"{training.id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(training), fh, indent=2)
