"""Configuration for the Federation Manager service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8015


settings = Settings()
