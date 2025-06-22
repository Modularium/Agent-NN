"""Settings for the Coalition Manager."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8012


settings = Settings()
