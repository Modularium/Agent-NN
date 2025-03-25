"""Configuration for Smolitux UI integration."""
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel

class SmolituxConfig(BaseModel):
    """Configuration for Smolitux UI integration."""
    
    # API Configuration
    api_prefix: str = "/smolitux"
    api_version: str = "v1"
    
    # UI Configuration
    default_language: str = "de"
    available_languages: list = ["de", "en"]
    
    # Theme Configuration
    default_theme: str = "light"
    available_themes: list = ["light", "dark"]
    
    # Agent Configuration
    default_agent: Optional[str] = None
    
    # LLM Configuration
    llm_backend: str = os.getenv("LLM_BACKEND", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    
    # System Configuration
    enable_logging: bool = True
    enable_metrics: bool = True
    
    # UI Components
    enable_chat: bool = True
    enable_agents_view: bool = True
    enable_tasks_view: bool = True
    enable_settings: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return self.dict()
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SmolituxConfig":
        """Create from dictionary.
        
        Args:
            data: Configuration data
            
        Returns:
            SmolituxConfig: Configuration instance
        """
        return cls(**data)
        
    @classmethod
    def from_env(cls) -> "SmolituxConfig":
        """Create from environment variables.
        
        Returns:
            SmolituxConfig: Configuration instance
        """
        return cls(
            default_language=os.getenv("SMOLITUX_DEFAULT_LANGUAGE", "de"),
            default_theme=os.getenv("SMOLITUX_DEFAULT_THEME", "light"),
            default_agent=os.getenv("SMOLITUX_DEFAULT_AGENT"),
            llm_backend=os.getenv("LLM_BACKEND", "openai"),
            llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            enable_logging=os.getenv("SMOLITUX_ENABLE_LOGGING", "true").lower() == "true",
            enable_metrics=os.getenv("SMOLITUX_ENABLE_METRICS", "true").lower() == "true",
            enable_chat=os.getenv("SMOLITUX_ENABLE_CHAT", "true").lower() == "true",
            enable_agents_view=os.getenv("SMOLITUX_ENABLE_AGENTS_VIEW", "true").lower() == "true",
            enable_tasks_view=os.getenv("SMOLITUX_ENABLE_TASKS_VIEW", "true").lower() == "true",
            enable_settings=os.getenv("SMOLITUX_ENABLE_SETTINGS", "true").lower() == "true"
        )

# Default configuration
default_config = SmolituxConfig()

# Get configuration from environment
config = SmolituxConfig.from_env()