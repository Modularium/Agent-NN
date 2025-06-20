from fastapi import APIRouter
from pydantic import BaseModel

from .service import SessionManagerService

router = APIRouter()
service = SessionManagerService()


class SessionCreate(BaseModel):
    data: dict


@router.post("/session")
async def create_session(payload: SessionCreate) -> dict:
    sid = service.create_session(payload.data)
    return {"session_id": sid}


@router.get("/session/{sid}")
async def get_session(sid: str) -> dict:
    data = service.get_session(sid)
    return {"data": data}


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
