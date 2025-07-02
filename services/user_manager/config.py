"""Configuration for the User Manager service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8005
    users_file: str = "users.json"


settings = Settings()
