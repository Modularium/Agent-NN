"""Configuration for the Vector Store service."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8004


settings = Settings()
