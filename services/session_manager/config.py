"""Configuration for the Session Manager service."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8005


settings = Settings()
