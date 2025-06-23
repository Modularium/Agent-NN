"""Config for Critic Agent service."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8110

settings = Settings()
