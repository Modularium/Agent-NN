from __future__ import annotations

"""Privacy-related utilities and access level definitions."""

from enum import Enum


class AccessLevel(str, Enum):
    """Data access classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
