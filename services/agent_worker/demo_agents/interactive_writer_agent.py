"""Writer agent with live feedback via AgentBus."""

from __future__ import annotations

from datetime import datetime
import time

from core.agent_bus import publish, subscribe
from core.model_context import ModelContext

from .writer_agent import WriterAgent


class InteractiveWriterAgent(WriterAgent):
    """Writer agent that requests critic feedback and applies it."""

    def run(self, ctx: ModelContext) -> ModelContext:  # type: ignore[override]
        ctx = super().run(ctx)
        publish(
            "critic_agent",
            {
                "sender": "interactive_writer_agent",
                "receiver": "critic_agent",
                "type": "request",
                "payload": {"text": ctx.result},
                "timestamp": datetime.now().isoformat(),
            },
        )
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            for msg in subscribe("interactive_writer_agent"):
                if msg.get("type") == "feedback":
                    comment = msg.get("payload", {}).get("comment", "")
                    if comment:
                        ctx.result = f"{ctx.result}\n{comment}"
                    ctx.metrics = ctx.metrics or {}
                    ctx.metrics["iterations"] = 2
                    return ctx
            time.sleep(0.05)
        return ctx
