"""Simple Q-learning trainer for agent selection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from core.model_context import TaskContext


@dataclass
class QTableLearner:
    """Q-learning based policy for choosing agents."""

    learning_rate: float = 0.1
    discount: float = 0.9
    epsilon: float = 0.1
    table: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def _state_key(self, task: TaskContext) -> str:
        return task.task_type

    def select_agent(self, task: TaskContext, agents: list[str]) -> str:
        """Choose an agent using an epsilon-greedy policy."""
        import random

        if random.random() < self.epsilon:
            return random.choice(agents)
        state = self._state_key(task)
        q_vals = {a: self.table.get((state, a), 0.0) for a in agents}
        return max(q_vals, key=q_vals.get)

    def learn(self, task: TaskContext, agent: str, reward: float) -> None:
        """Update Q-table for given state/action pair."""
        state = self._state_key(task)
        key = (state, agent)
        current = self.table.get(key, 0.0)
        self.table[key] = current + self.learning_rate * (reward - current)

    def save(self, path: str) -> None:
        data = {"table": {f"{s}|{a}": v for (s, a), v in self.table.items()}}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load(self, path: str) -> None:
        file = Path(path)
        if not file.exists():
            return
        with open(file, encoding="utf-8") as fh:
            data = json.load(fh)
        table: Dict[Tuple[str, str], float] = {}
        for key, val in data.get("table", {}).items():
            state, agent = key.split("|", 1)
            table[(state, agent)] = float(val)
        self.table = table
