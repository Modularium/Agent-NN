from __future__ import annotations

import os
from typing import Dict

import yaml
from pydantic_settings import BaseSettings

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .local_provider import GGUFProvider, LocalHFProvider
from .openai_provider import OpenAIProvider


class LLMConfig(BaseSettings):
    default_provider: str = "openai"
    providers: Dict[str, Dict] = {}

    @classmethod
    def load(cls) -> "LLMConfig":
        path = os.getenv(
            "LLM_CONFIG_PATH", os.path.join(os.getcwd(), "llm_config.yaml")
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()


class LLMBackendManager:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.load()
        self._cache: Dict[str, LLMProvider] = {}

    def get_provider(self, name: str | None = None) -> LLMProvider:
        provider_name = name or self.config.default_provider
        if provider_name in self._cache:
            return self._cache[provider_name]
        info = self.config.providers.get(provider_name)
        if not info:
            raise ValueError(f"unknown provider {provider_name}")
        type_ = info.get("type")
        if type_ == "openai":
            provider = OpenAIProvider(
                api_key=os.getenv("OPENAI_API_KEY", info.get("api_key"))
            )
        elif type_ == "anthropic":
            provider = AnthropicProvider(
                api_key=os.getenv("ANTHROPIC_API_KEY", info.get("api_key"))
            )
        elif type_ == "local":
            provider = LocalHFProvider(model_path=info.get("model_path", ""))
        elif type_ == "gguf":
            provider = GGUFProvider(model_path=info.get("model_path", ""))
        else:
            raise ValueError(f"unsupported provider type {type_}")
        self._cache[provider_name] = provider
        return provider

    def available_models(self) -> Dict[str, Dict]:
        return self.config.providers
