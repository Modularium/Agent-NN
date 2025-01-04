"""LLM backend manager for handling different LLM providers."""
import os
from enum import Enum
from typing import Dict, Any, Optional
import yaml
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from .lmstudio_backend import LMStudioLLM, LMStudioEmbeddings, create_llm as create_lmstudio_llm, create_embeddings as create_lmstudio_embeddings

class LLMBackendType(Enum):
    """Supported LLM backend types."""
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"
    LMSTUDIO = "lmstudio"

class LLMBackendManager:
    """Manager for LLM backends."""
    
    def __init__(self, backend_type: LLMBackendType = LLMBackendType.LMSTUDIO):
        self.backend_type = backend_type
        self.config = self._load_config()
        self.llm = None
        self.embeddings = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load LLM configuration from YAML or JSON file."""
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "llm_config.yaml")
        try:
            with open(config_path, "r") as f:
                content = f.read()
                try:
                    config = yaml.safe_load(content)
                except:
                    config = json.loads(content)
                if config is None:
                    config = {"llm": {}}
                return config.get(self.backend_type.value, config["llm"])
        except (FileNotFoundError, KeyError):
            return {}
    
    def get_llm(self, **kwargs: Any) -> Any:
        """Get LLM instance."""
        if self.llm is None:
            if self.backend_type == LLMBackendType.OPENAI:
                self.llm = self._create_openai_llm(**kwargs)
            elif self.backend_type == LLMBackendType.AZURE:
                self.llm = self._create_azure_llm(**kwargs)
            elif self.backend_type == LLMBackendType.LMSTUDIO:
                self.llm = create_lmstudio_llm()
            else:
                raise ValueError(f"Unsupported backend type: {self.backend_type}")
        return self.llm
    
    def get_embeddings(self, **kwargs: Any) -> Any:
        """Get embeddings instance."""
        if self.embeddings is None:
            if self.backend_type == LLMBackendType.OPENAI:
                self.embeddings = self._create_openai_embeddings(**kwargs)
            elif self.backend_type == LLMBackendType.AZURE:
                self.embeddings = self._create_azure_embeddings(**kwargs)
            elif self.backend_type == LLMBackendType.LMSTUDIO:
                self.embeddings = create_lmstudio_embeddings()
            else:
                raise ValueError(f"Unsupported backend type: {self.backend_type}")
        return self.embeddings
    
    def _create_openai_llm(self, **kwargs: Any) -> Any:
        """Create OpenAI LLM instance."""
        config = {
            "model_name": self.config.get("model", "gpt-3.5-turbo"),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 2048),
            "top_p": self.config.get("top_p", 0.95),
            "frequency_penalty": self.config.get("frequency_penalty", 0.0),
            "presence_penalty": self.config.get("presence_penalty", 0.0),
            "stop": self.config.get("stop", None),
            **kwargs
        }
        
        if config["model_name"].startswith("gpt-3.5") or config["model_name"].startswith("gpt-4"):
            return ChatOpenAI(**config)
        return OpenAI(**config)
    
    def _create_azure_llm(self, **kwargs: Any) -> Any:
        """Create Azure LLM instance."""
        config = {
            "deployment_name": self.config["deployment_name"],
            "model_name": self.config.get("model", "gpt-3.5-turbo"),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 2048),
            "top_p": self.config.get("top_p", 0.95),
            "frequency_penalty": self.config.get("frequency_penalty", 0.0),
            "presence_penalty": self.config.get("presence_penalty", 0.0),
            "stop": self.config.get("stop", None),
            **kwargs
        }
        
        if config["model_name"].startswith("gpt-3.5") or config["model_name"].startswith("gpt-4"):
            return ChatOpenAI(**config)
        return OpenAI(**config)
    
    def _create_openai_embeddings(self, **kwargs: Any) -> Any:
        """Create OpenAI embeddings instance."""
        config = {
            "model": self.config.get("embedding_model", "text-embedding-ada-002"),
            **kwargs
        }
        return OpenAIEmbeddings(**config)
    
    def _create_azure_embeddings(self, **kwargs: Any) -> Any:
        """Create Azure embeddings instance."""
        config = {
            "deployment": self.config["embedding_deployment"],
            "model": self.config.get("embedding_model", "text-embedding-ada-002"),
            **kwargs
        }
        return OpenAIEmbeddings(**config)
