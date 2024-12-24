"""LLM backend management for different model providers."""
import os
import subprocess
from typing import Dict, Any, Optional, List
import requests
from enum import Enum
from langchain.llms.base import LLM
from langchain.llms import OpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from utils.logging_util import setup_logger

logger = setup_logger(__name__)

class LLMBackendType(Enum):
    """Supported LLM backend types."""
    OPENAI = "openai"
    LMSTUDIO = "lmstudio"
    LLAMAFILE = "llamafile"

class LMStudioLLM(LLM):
    """LangChain integration for LM Studio."""
    
    def __init__(self,
                 endpoint_url: str = "http://localhost:1234/v1",
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        """Initialize LM Studio LLM.
        
        Args:
            endpoint_url: URL of the LM Studio API endpoint
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__()
        self.endpoint_url = endpoint_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        """Call LM Studio API.
        
        Args:
            prompt: Input prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments
            
        Returns:
            str: Generated text
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": stop or []
        }
        
        try:
            response = requests.post(
                f"{self.endpoint_url}/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error calling LM Studio API: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "lmstudio"

class LlamafileLLM(LLM):
    """LangChain integration for Llamafile."""
    
    def __init__(self,
                 model_path: str,
                 port: int = 8080,
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        """Initialize Llamafile LLM.
        
        Args:
            model_path: Path to the Llamafile executable
            port: Port to run the server on
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__()
        self.model_path = model_path
        self.port = port
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.process = None
        self.endpoint_url = f"http://localhost:{port}/v1"
        
    def start_server(self):
        """Start the Llamafile server."""
        if self.process is None:
            try:
                # Make the Llamafile executable
                os.chmod(self.model_path, 0o755)
                
                # Start the server
                self.process = subprocess.Popen(
                    [self.model_path, "--server", "--port", str(self.port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                logger.info(f"Started Llamafile server on port {self.port}")
            except Exception as e:
                logger.error(f"Error starting Llamafile server: {str(e)}")
                raise
                
    def stop_server(self):
        """Stop the Llamafile server."""
        if self.process is not None:
            self.process.terminate()
            self.process = None
            logger.info("Stopped Llamafile server")
            
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        """Call Llamafile API.
        
        Args:
            prompt: Input prompt
            stop: Optional stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments
            
        Returns:
            str: Generated text
        """
        if self.process is None:
            self.start_server()
            
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": stop or []
        }
        
        try:
            response = requests.post(
                f"{self.endpoint_url}/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error calling Llamafile API: {str(e)}")
            raise
            
    def __del__(self):
        """Clean up resources."""
        self.stop_server()

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llamafile"

class LLMBackendManager:
    """Manager for different LLM backends."""
    
    def __init__(self):
        """Initialize the LLM backend manager."""
        self.current_backend = LLMBackendType.OPENAI
        self.llamafile_process = None
        self.models: Dict[str, Dict[str, Any]] = {
            "openai": {
                "gpt-3.5-turbo": {"max_tokens": 2048, "temperature": 0.7},
                "gpt-4": {"max_tokens": 4096, "temperature": 0.7}
            },
            "lmstudio": {
                "local-model": {"max_tokens": 1024, "temperature": 0.7}
            },
            "llamafile": {
                "llama-2-7b": {
                    "path": "/path/to/llama-2-7b.llamafile",
                    "max_tokens": 1024,
                    "temperature": 0.7
                }
            }
        }
        
    def get_llm(self,
                backend_type: Optional[LLMBackendType] = None,
                model_name: Optional[str] = None,
                **kwargs) -> LLM:
        """Get an LLM instance.
        
        Args:
            backend_type: Type of backend to use
            model_name: Name of the model to use
            **kwargs: Additional model parameters
            
        Returns:
            LLM: LangChain LLM instance
        """
        backend_type = backend_type or self.current_backend
        
        if backend_type == LLMBackendType.OPENAI:
            model_name = model_name or "gpt-3.5-turbo"
            model_config = self.models["openai"][model_name].copy()
            model_config.update(kwargs)
            return OpenAI(
                model_name=model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                **model_config
            )
            
        elif backend_type == LLMBackendType.LMSTUDIO:
            model_config = self.models["lmstudio"]["local-model"].copy()
            model_config.update(kwargs)
            return LMStudioLLM(**model_config)
            
        elif backend_type == LLMBackendType.LLAMAFILE:
            model_name = model_name or "llama-2-7b"
            model_config = self.models["llamafile"][model_name].copy()
            model_config.update(kwargs)
            return LlamafileLLM(**model_config)
            
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
            
    def set_backend(self, backend_type: LLMBackendType):
        """Set the default backend type.
        
        Args:
            backend_type: Type of backend to use
        """
        self.current_backend = backend_type
        logger.info(f"Set default backend to {backend_type.value}")
        
    def add_model(self,
                 backend_type: LLMBackendType,
                 model_name: str,
                 model_config: Dict[str, Any]):
        """Add a new model configuration.
        
        Args:
            backend_type: Type of backend
            model_name: Name of the model
            model_config: Model configuration
        """
        self.models[backend_type.value][model_name] = model_config
        logger.info(f"Added model {model_name} for backend {backend_type.value}")
        
    def get_available_models(self,
                           backend_type: Optional[LLMBackendType] = None) -> Dict[str, List[str]]:
        """Get available models.
        
        Args:
            backend_type: Optional backend type to filter by
            
        Returns:
            Dict mapping backend types to lists of model names
        """
        if backend_type:
            return {backend_type.value: list(self.models[backend_type.value].keys())}
        return {k: list(v.keys()) for k, v in self.models.items()}