"""API routes for the Session Manager service."""

from fastapi import APIRouter

from core.model_context import ModelContext
from core.schemas import StatusResponse
from utils.api_utils import api_route

from .schemas import ModelSelection, SessionHistory, SessionId
from .service import SessionManagerService

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
    return StatusResponse(status="ok")


@api_route(version="v1.0.0")
@router.get("/model/{user_id}", response_model=ModelSelection)
async def get_model(user_id: str) -> ModelSelection:
    model = service.get_model(user_id) or ""
    return ModelSelection(user_id=user_id, model_id=model)
