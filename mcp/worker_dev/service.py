"""Developer worker executing simple code generation tasks."""

from __future__ import annotations

import json
import os
import urllib.request
import logging


class WorkerService:
    """Interact with the LLM gateway to fulfil dev tasks."""

    def __init__(self) -> None:
        self.llm_url = os.getenv("LLM_GATEWAY_URL", "http://llm_gateway:8000")
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def _call_llm(self, prompt: str) -> str:
        url = self.llm_url.rstrip("/") + "/generate"
        payload = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            self.logger.info("LLM request to %s", url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("text", "")
        except Exception as exc:  # pragma: no cover - network errors
            self.logger.error("LLM request failed: %s", exc)
            return f"error: {exc}"

    def execute_task(self, task: str) -> str:
        prompt = f"Write a Python function that {task}."
        return self._call_llm(prompt)
