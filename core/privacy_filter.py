from __future__ import annotations

"""Utilities to filter context data according to access levels."""

from typing import Tuple

from .model_context import ModelContext, AccessText
from .privacy import AccessLevel


_LEVEL_ORDER = [
    AccessLevel.PUBLIC,
    AccessLevel.INTERNAL,
    AccessLevel.CONFIDENTIAL,
    AccessLevel.SENSITIVE,
]

_index = {level: i for i, level in enumerate(_LEVEL_ORDER)}


def _needs_redaction(level: AccessLevel, max_access: AccessLevel) -> bool:
    return _index[level] > _index[max_access]


def redact_context(ctx: ModelContext, max_access: AccessLevel) -> ModelContext:
    """Return a copy of ``ctx`` with data above ``max_access`` redacted."""

    new_ctx = ctx.model_copy(deep=True)
    redacted = 0

    if new_ctx.task_context and new_ctx.task_context.input_data:
        item = new_ctx.task_context.input_data
        if _needs_redaction(item.access, max_access):
            item.text = "[REDACTED]"
            redacted += 1

    if new_ctx.task_context and new_ctx.task_context.description:
        desc = new_ctx.task_context.description
        if _needs_redaction(desc.access, max_access):
            desc.text = "[REDACTED]"
            redacted += 1

    if new_ctx.memory:
        for m in new_ctx.memory:
            if _needs_redaction(m.access, max_access):
                m.text = "[REDACTED]"
                redacted += 1

    new_ctx.metrics = new_ctx.metrics or {}
    new_ctx.metrics["context_redacted_fields"] = redacted
    return new_ctx
