"""Configuration for the LLM Gateway service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8003


settings = Settings()
