"""Configuration for the Agent Registry service."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8002


settings = Settings()
