from __future__ import annotations

import httpx


def request_routing(task_type: str, routing_url: str) -> str | None:
    """Return target worker for task_type by calling the routing service."""
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{routing_url.rstrip('/')}/route",
                json={"task_type": task_type},
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json().get("target_worker")
    except Exception:  # pragma: no cover - network error
        return None
