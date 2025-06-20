"""API routes for Session Manager service."""

from fastapi import APIRouter

from .schemas import SessionData
from .service import SessionManagerService

router = APIRouter()
service = SessionManagerService()


@router.get("/session/{session_id}")
async def get_session(session_id: str) -> SessionData | None:
    """Fetch a session."""
    data = service.get_session(session_id)
    return SessionData(data=data) if data else None


@router.post("/session/{session_id}")
async def save_session(session_id: str, payload: SessionData) -> dict:
    """Persist a session."""
    service.save_session(session_id, payload.data)
    return {"status": "saved"}
