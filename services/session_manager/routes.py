"""API routes for the Session Manager service."""

from fastapi import APIRouter

from core.model_context import ModelContext
from core.schemas import StatusResponse
from utils.api_utils import api_route

from .schemas import (
    ModelSelection,
    SessionHistory,
    SessionId,
    Feedback,
    FeedbackList,
)
from .service import SessionManagerService
from core.audit import audit_action
from core.feedback_utils import FeedbackEntry

router = APIRouter()
service = SessionManagerService()


@api_route(version="v1.0.0")
@router.post("/start_session", response_model=SessionId)
async def start_session() -> SessionId:
    """Create a new session and return its id."""
    sid = service.start_session()
    return SessionId(session_id=sid)


@api_route(version="v1.0.0")
@router.post("/update_context", response_model=StatusResponse)
async def update_context(ctx: ModelContext) -> StatusResponse:
    """Store or extend a session context."""
    service.update_context(ctx)
    return StatusResponse(status="ok")


@api_route(version="v1.0.0")
@router.get("/context/{session_id}", response_model=SessionHistory)
async def get_context(session_id: str) -> SessionHistory:
    """Return the conversation history for a session."""
    history = service.get_context(session_id)
    return SessionHistory(context=history)


@api_route(version="v1.0.0")
@router.post("/model", response_model=StatusResponse)
async def set_model(selection: ModelSelection) -> StatusResponse:
    service.set_model(selection.user_id, selection.model_id)
    audit_action(
        actor=selection.user_id,
        action="set_model",
        context_id=selection.user_id,
        detail={"model": selection.model_id},
    )
    return StatusResponse(status="ok")


@api_route(version="v1.0.0")
@router.get("/model/{user_id}", response_model=ModelSelection)
async def get_model(user_id: str) -> ModelSelection:
    model = service.get_model(user_id) or ""
    return ModelSelection(user_id=user_id, model_id=model)


@api_route(version="v1.0.0")
@router.post("/session/{session_id}/feedback", response_model=StatusResponse)
async def add_feedback(session_id: str, fb: Feedback) -> StatusResponse:
    entry = FeedbackEntry(
        session_id=session_id,
        user_id=fb.user_id,
        agent_id=fb.agent_id,
        score=fb.score,
        comment=fb.comment,
        timestamp=fb.timestamp,
    )
    service.add_feedback(entry)
    audit_action(
        actor=fb.user_id,
        action="feedback_submit",
        context_id=session_id,
        detail={"score": fb.score, "agent_id": fb.agent_id},
    )
    return StatusResponse(status="ok")


@api_route(version="v1.0.0")
@router.get("/session/{session_id}/feedback", response_model=FeedbackList)
async def get_feedback(session_id: str) -> FeedbackList:
    data = service.get_feedback(session_id)
    return FeedbackList(items=data)


@api_route(version="v1.0.0")
@router.get("/feedback/stats", response_model=dict)
async def feedback_stats() -> dict:
    return service.get_feedback_stats()
