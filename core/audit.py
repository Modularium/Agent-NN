from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

from .audit_log import AuditEntry, AuditLog

_audit = AuditLog(log_dir=os.getenv("AUDIT_LOG_DIR", "audit"))


def audit_action(
    actor: str,
    action: str,
    context_id: str,
    detail: Dict[str, Any],
    signature: Optional[str] = None,
) -> str:
    """Write an audit entry and return its id."""
    entry = AuditEntry(
        timestamp=datetime.utcnow().isoformat(),
        actor=actor,
        action=action,
        context_id=context_id,
        detail=detail,
        signature=signature,
    )
    return _audit.write(entry)
