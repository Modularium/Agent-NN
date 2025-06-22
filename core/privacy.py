"""Privacy-related utilities and access level definitions."""

from __future__ import annotations

from enum import Enum


class AccessLevel(str, Enum):
    """Data access classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
