from pydantic import BaseModel


class RegisterRequest(BaseModel):
    username: str
    password: str


class RegisterResponse(BaseModel):
    success: bool


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ValidateRequest(BaseModel):
    token: str
