"""LLM backend manager for handling different LLM providers."""
import os
from enum import Enum
from typing import Dict, Any, Optional, List
import yaml
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
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
        self._backend_type = backend_type
        self.config = self._load_config()
        self.llm = None
        self.embeddings = None
        
    @property
    def current_backend(self) -> LLMBackendType:
        """Get current backend type."""
        return self._backend_type
        
    def set_backend(self, backend_type: LLMBackendType):
        """Set backend type."""
        self._backend_type = backend_type
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
                return config.get(self._backend_type.value, config["llm"])
        except (FileNotFoundError, KeyError):
            return {}
    
    def get_llm(self, **kwargs: Any) -> Any:
        """Get LLM instance."""
        if self.llm is None:
            if self._backend_type == LLMBackendType.OPENAI:
                self.llm = self._create_openai_llm(**kwargs)
            elif self._backend_type == LLMBackendType.AZURE:
                self.llm = self._create_azure_llm(**kwargs)
            elif self._backend_type == LLMBackendType.LMSTUDIO:
                try:
                    self.llm = create_lmstudio_llm()
                except Exception as e:
                    print(f"Warning: Failed to connect to LM-Studio: {e}")
                    print("Using mock LLM for testing...")
                    from langchain.llms.fake import FakeListLLM
                    self.llm = FakeListLLM(responses=["This is a mock response from LM-Studio."])
            else:
                raise ValueError(f"Unsupported backend type: {self._backend_type}")
        return self.llm
    
    def get_embeddings(self, **kwargs: Any) -> Any:
        """Get embeddings instance."""
        if self.embeddings is None:
            if self._backend_type == LLMBackendType.OPENAI:
                self.embeddings = self._create_openai_embeddings(**kwargs)
            elif self._backend_type == LLMBackendType.AZURE:
                self.embeddings = self._create_azure_embeddings(**kwargs)
            elif self._backend_type == LLMBackendType.LMSTUDIO:
                try:
                    self.embeddings = create_lmstudio_embeddings()
                except Exception as e:
                    print(f"Warning: Failed to connect to LM-Studio for embeddings: {e}")
                    print("Using HuggingFace embeddings as fallback...")
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2",
                        model_kwargs={"device": "cpu"}
                    )
            else:
                raise ValueError(f"Unsupported backend type: {self._backend_type}")
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
        
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for each backend type.
        
        Returns:
            Dict mapping backend types to lists of available model names
        """
        return {
            LLMBackendType.OPENAI.value: ["gpt-3.5-turbo", "gpt-4"],
            LLMBackendType.LMSTUDIO.value: ["local-model"],
            LLMBackendType.AZURE.value: ["gpt-3.5-turbo", "gpt-4"],
            LLMBackendType.LOCAL.value: ["local-model"]
        }
