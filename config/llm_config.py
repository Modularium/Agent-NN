"""Configuration for LLM backends."""
import os
from typing import Dict, Any

# Default paths for local models
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)

# LM Studio configuration
LMSTUDIO_CONFIG = {
    "endpoint_url": os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1"),
    "models": {
        "local-model": {
            "max_tokens": 1024,
            "temperature": 0.7,
            "description": "Default local model in LM Studio"
        }
    }
}

# Llamafile configuration
LLAMAFILE_CONFIG = {
    "models_dir": os.getenv("LLAMAFILE_MODELS_DIR", DEFAULT_MODELS_DIR),
    "models": {
        "llama-2-7b": {
            "filename": "llama-2-7b.gguf",
            "max_tokens": 1024,
            "temperature": 0.7,
            "description": "Llama 2 7B base model"
        },
        "llama-2-13b": {
            "filename": "llama-2-13b.gguf",
            "max_tokens": 2048,
            "temperature": 0.7,
            "description": "Llama 2 13B base model"
        },
        "codellama-7b": {
            "filename": "codellama-7b.gguf",
            "max_tokens": 1024,
            "temperature": 0.7,
            "description": "CodeLlama 7B for technical tasks"
        }
    }
}

# OpenAI configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "models": {
        "gpt-3.5-turbo": {
            "max_tokens": 2048,
            "temperature": 0.7,
            "description": "GPT-3.5 Turbo for general tasks"
        },
        "gpt-4": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "description": "GPT-4 for complex tasks"
        }
    }
}

# Domain-specific model preferences
DOMAIN_MODEL_PREFERENCES = {
    "finance": {
        "openai": "gpt-4",  # More precise for financial calculations
        "lmstudio": "local-model",
        "llamafile": "llama-2-13b"  # Larger model for complex tasks
    },
    "tech": {
        "openai": "gpt-3.5-turbo",
        "lmstudio": "local-model",
        "llamafile": "codellama-7b"  # Specialized for code
    },
    "marketing": {
        "openai": "gpt-3.5-turbo",
        "lmstudio": "local-model",
        "llamafile": "llama-2-7b"
    }
}

def get_model_path(model_name: str) -> str:
    """Get the full path for a Llamafile model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        str: Full path to the model file
    """
    if model_name not in LLAMAFILE_CONFIG["models"]:
        raise ValueError(f"Unknown model: {model_name}")
        
    return os.path.join(
        LLAMAFILE_CONFIG["models_dir"],
        LLAMAFILE_CONFIG["models"][model_name]["filename"]
    )

def get_model_config(backend: str, model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model.
    
    Args:
        backend: Backend type (openai, lmstudio, or llamafile)
        model_name: Name of the model
        
    Returns:
        Dict containing model configuration
    """
    configs = {
        "openai": OPENAI_CONFIG,
        "lmstudio": LMSTUDIO_CONFIG,
        "llamafile": LLAMAFILE_CONFIG
    }
    
    if backend not in configs:
        raise ValueError(f"Unknown backend: {backend}")
        
    config = configs[backend]
    if model_name not in config["models"]:
        raise ValueError(f"Unknown model {model_name} for backend {backend}")
        
    return config["models"][model_name]

def get_preferred_model(domain: str, backend: str) -> str:
    """Get the preferred model for a domain and backend.
    
    Args:
        domain: Domain name
        backend: Backend type
        
    Returns:
        str: Name of the preferred model
    """
    if domain not in DOMAIN_MODEL_PREFERENCES:
        raise ValueError(f"Unknown domain: {domain}")
        
    if backend not in DOMAIN_MODEL_PREFERENCES[domain]:
        raise ValueError(f"Unknown backend {backend} for domain {domain}")
        
    return DOMAIN_MODEL_PREFERENCES[domain][backend]