from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import json
import os
import httpx

from .agent_profile import AgentIdentity

PROMPT_FILE = Path("prompts/evolve_profile_prompt.txt")


def _load_prompt() -> str:
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8")
    return """Given the following agent profile and interaction history, propose an updated skill list and traits:\nProfile: {agent_profile}\nHistory: {recent_interactions}"""


def _call_llm(prompt: str) -> Dict[str, Any] | None:
    url = os.getenv("LLM_GATEWAY_URL", "http://localhost:8003").rstrip("/") + "/generate"
    payload = {"prompt": prompt, "temperature": 0.3, "max_tokens": 512}
    try:
        with httpx.Client() as client:
            resp = client.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("completion", "")
            return json.loads(data)
    except Exception:
        return None


def _heuristic_update(agent: AgentIdentity, history: List[Dict[str, Any]]) -> AgentIdentity:
    ratings = [1 if h.get("rating") == "good" else 0 for h in history if "rating" in h]
    if ratings:
        pos = sum(ratings)
        total = len(ratings)
        agent.traits["precision"] = round(pos / total, 3)
        agent.traits["harshness"] = round(1 - (pos / total), 3)
    return agent


def evolve_profile(agent: AgentIdentity, history: List[Dict[str, Any]], mode: str = "llm") -> AgentIdentity:
    """Return an updated agent profile based on history."""
    history = history or []
    if mode == "heuristic":
        return _heuristic_update(agent, history)

    if mode == "llm":
        prompt_template = _load_prompt()
        prompt = prompt_template.format(
            agent_profile=json.dumps(asdict(agent)),
            recent_interactions=json.dumps(history[-20:]),
        )
        result = _call_llm(prompt)
        if isinstance(result, dict):
            traits = result.get("traits")
            if isinstance(traits, dict):
                agent.traits.update(traits)
            skills = result.get("skills")
            if isinstance(skills, list):
                agent.skills = list({*agent.skills, *skills})
        return agent

    raise ValueError(f"unknown mode: {mode}")
