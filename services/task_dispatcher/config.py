"""Configuration for the Task Dispatcher service."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8001


settings = Settings()
