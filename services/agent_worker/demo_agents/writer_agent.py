"""Simple writer agent using the LLM gateway."""

from __future__ import annotations

import httpx
from typing import Any

from core.model_context import ModelContext
from core.metrics_utils import TASKS_PROCESSED, TOKENS_IN, TOKENS_OUT
from core.agent_profile import AgentIdentity
from core.agent_evolution import evolve_profile
import os
import json
from pathlib import Path
from datetime import datetime


class WriterAgent:
    """Generate a text completion from a prompt."""

    def __init__(self, llm_url: str = "http://localhost:8003") -> None:
        self.llm_url = llm_url.rstrip("/")
        self.profile = AgentIdentity.load("writer_agent")
        self._history: list[dict[str, Any]] = []
        self._runs = 0
        self._evolve_enabled = os.getenv("AGENT_EVOLVE", "false").lower() == "true"
        self._evolve_interval = int(os.getenv("AGENT_EVOLVE_INTERVAL", "5"))
        self._evolve_mode = os.getenv("AGENT_EVOLVE_MODE", "heuristic")
        self._log_dir = Path(os.getenv("AGENT_LOG_DIR", "agent_log"))

    def run(self, ctx: ModelContext) -> ModelContext:
        prompt = ctx.task_context.description or ""
        if ctx.memory:
            recent = [m.get("output", "") for m in ctx.memory[-2:] if isinstance(m.get("output"), str)]
            if recent:
                prompt = " ".join(recent) + "\n" + prompt
        if ctx.task_context.input_data:
            prompt = f"{prompt}\n{ctx.task_context.input_data}"
        if isinstance(ctx.result, list):
            docs = "\n".join(d.get("text", "") for d in ctx.result)
            prompt = f"{prompt}\n\n{docs}" if docs else prompt
        TOKENS_IN.labels("writer_agent").inc(len(prompt.split()))
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{self.llm_url}/generate",
                    json={"prompt": prompt},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            data = {"completion": prompt, "tokens_used": 0}
        TOKENS_OUT.labels("writer_agent").inc(data.get("tokens_used", 0))
        TASKS_PROCESSED.labels("writer_agent").inc()
        ctx.result = data.get("completion")
        ctx.metrics = {"tokens_used": data.get("tokens_used", 0)}
        self._runs += 1
        self._history.append({"rating": "good"})
        if self._evolve_enabled and self._runs % self._evolve_interval == 0:
            self.profile = evolve_profile(self.profile, self._history, self._evolve_mode)
            self.profile.save()
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._log_dir / f"{self.profile.name}.log"
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"timestamp": datetime.now().isoformat(), "traits": self.profile.traits, "skills": self.profile.skills}) + "\n")
        return ctx
