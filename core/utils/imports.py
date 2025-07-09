"""Helper utilities for optional imports used across the project."""
from __future__ import annotations

from importlib import import_module
from typing import Any


def optional_import(name: str) -> Any | None:
    """Return imported module if available, otherwise ``None``."""
    try:
        return import_module(name)
    except ImportError:  # pragma: no cover - optional dependency
        return None


# common optional deps
torch = optional_import("torch")

__all__ = ["optional_import", "torch"]
