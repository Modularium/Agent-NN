from fastapi import APIRouter
from pydantic import BaseModel

from .service import SessionManagerService

router = APIRouter()
service = SessionManagerService()


class SessionCreate(BaseModel):
    data: dict


class Feedback(BaseModel):
    rating: str
    comment: str | None = None
    index: int


@router.post("/session")
async def create_session(payload: SessionCreate) -> dict:
    sid = service.create_session(payload.data)
    return {"session_id": sid}


@router.get("/session/{sid}")
async def get_session(sid: str) -> dict:
    data = service.get_session(sid)
    return {"data": data}


@router.get("/session/{sid}/history")
async def get_history(sid: str) -> dict:
    return {"history": service.get_history(sid)}


@router.post("/session/{sid}/feedback")
async def post_feedback(sid: str, feedback: Feedback) -> dict:
    service.add_feedback(
        sid,
        {
            "index": feedback.index,
            "rating": feedback.rating,
            "comment": feedback.comment,
        },
    )
    return {"status": "ok"}


@router.get("/session/{sid}/status")
async def session_status(sid: str) -> dict:
    data = service.get_session(sid)
    return {"status": "active" if data is not None else "expired"}


@router.get("/sessions")
async def list_sessions() -> dict:
    return {"sessions": list(service._sessions.keys())}


@router.get("/feedback/stats")
async def feedback_stats() -> dict:
    return service.get_feedback_stats()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
