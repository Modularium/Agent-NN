"""Config for Example Agent Worker."""

from pydantic import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8010


settings = Settings()
