"""API routes for the User Manager service."""

from fastapi import APIRouter, HTTPException
from utils.api_utils import api_route

from .schemas import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    TokenResponse,
    ValidateRequest,
)
from .service import UserManagerService

router = APIRouter()
service = UserManagerService()


@api_route(version="v1.0.0")
@router.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest) -> RegisterResponse:
    """Create a new user account."""
    if not service.create_user(req.username, req.password):
        raise HTTPException(status_code=409, detail="User already exists")
    return RegisterResponse(success=True)


@api_route(version="v1.0.0")
@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest) -> TokenResponse:
    """Authenticate user and return a token."""
    token = service.authenticate(req.username, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=token)


@api_route(version="v1.0.0")
@router.post("/validate")
async def validate(req: ValidateRequest) -> None:
    """Validate an authentication token."""
    if not service.validate_token(req.token):
        raise HTTPException(status_code=401, detail="Invalid token")
