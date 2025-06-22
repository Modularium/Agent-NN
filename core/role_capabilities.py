from __future__ import annotations

from typing import Any, Dict

from .model_context import ModelContext

ROLE_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "writer": {
        "max_tokens": 2000,
        "max_context_size": 5,
        "can_access_memory": True,
    },
    "retriever": {
        "max_tokens": 1500,
        "can_access_memory": True,
    },
    "critic": {
        "max_tokens": 1000,
        "can_modify_output": False,
    },
    "analyst": {
        "max_tokens": 1200,
        "max_context_size": 10,
        "can_access_memory": True,
    },
    "coordinator": {
        "max_tokens": 3000,
        "max_context_size": 20,
        "can_access_memory": True,
    },
}


def apply_role_capabilities(ctx: ModelContext, role: str) -> ModelContext:
    """Limit context fields according to ROLE_CAPABILITIES."""
    limits = ROLE_CAPABILITIES.get(role, {})
    ctx.applied_limits = limits
    if "max_tokens" in limits:
        if ctx.max_tokens is None or ctx.max_tokens > limits["max_tokens"]:
            ctx.max_tokens = limits["max_tokens"]
    if not limits.get("can_access_memory", True):
        ctx.memory = None
    if "max_context_size" in limits and ctx.memory is not None:
        if len(ctx.memory) > limits["max_context_size"]:
            ctx.memory = ctx.memory[-limits["max_context_size"] :]
    return ctx
