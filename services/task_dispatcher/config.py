"""Configuration for the Task Dispatcher service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8001
    registry_url: str = "http://localhost:8002"
    session_url: str = "http://localhost:8005"
    coordinator_url: str = "http://localhost:8010"


settings = Settings()
