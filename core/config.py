from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from ``.env``."""

    DATA_DIR: str = "data"
    SESSIONS_DIR: str = "data/sessions"
    VECTOR_DB_DIR: str = "data/vectorstore"
    LOG_DIR: str = "logs"
    MODELS_DIR: str = "models"
    EMBEDDINGS_CACHE_DIR: str = "embeddings_cache"
    EXPORT_DIR: str = "export"
    MEMORY_LOG_DIR: str = "data/memory_log"
    MEMORY_STORE_BACKEND: str = "memory"  # `memory` or `file`

    DEFAULT_STORE_BACKEND: str = "memory"  # ``memory`` or ``file``
    VECTOR_DB_BACKEND: str = "memory"  # ``memory`` or ``chromadb``

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

for path in [
    settings.DATA_DIR,
    settings.SESSIONS_DIR,
    settings.VECTOR_DB_DIR,
    settings.LOG_DIR,
    settings.MEMORY_LOG_DIR,
    settings.MODELS_DIR,
    settings.EMBEDDINGS_CACHE_DIR,
    settings.EXPORT_DIR,
]:
    Path(path).mkdir(parents=True, exist_ok=True)
