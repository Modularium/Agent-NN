"""Configuration for Routing Agent."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8111
    rules_path: str = "services/routing_agent/rules.yaml"
    meta_enabled: bool = False


settings = Settings()
