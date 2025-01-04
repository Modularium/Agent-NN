import os
from typing import Optional, Dict, Any
from langchain.llms.base import LLM
from langchain_community.llms import OpenAI
from langchain_community.llms import LlamaCpp
from config.llm_config import OPENAI_CONFIG, LLAMAFILE_CONFIG, get_model_path

class LocalLLM(LlamaCpp):
    def __init__(self, model_name: str = "llama-2-7b", **kwargs):
        config = LLAMAFILE_CONFIG["models"][model_name]
        super().__init__(
            model_path=get_model_path(model_name),
            temperature=kwargs.get("temperature", config.get("temperature", 0.7)),
            max_tokens=kwargs.get("max_tokens", config.get("max_tokens", 1024)),
            n_ctx=kwargs.get("n_ctx", 2048),
            n_batch=kwargs.get("n_batch", 512),
            n_gpu_layers=kwargs.get("n_gpu_layers", 0),
            verbose=kwargs.get("verbose", False)
        )
        
    def invoke(self, prompt: str, **kwargs) -> str:
        """Override invoke to handle empty responses."""
        response = super().invoke(prompt, **kwargs)
        if not response:
            return "I apologize, but I am unable to generate a response at this time."
        return response

class BaseLLM:
    def __init__(self, model_name: str = None, temperature: Optional[float] = None):
        """Initialize LLM with either OpenAI or Llamafile backend.
        
        Args:
            model_name: Name of the model to use. If None, defaults will be used based on backend
            temperature: Optional temperature parameter for generation
        """
        self.use_openai = bool(OPENAI_CONFIG["api_key"])
        
        if self.use_openai:
            if model_name is None:
                model_name = "gpt-3.5-turbo"
            config = OPENAI_CONFIG["models"][model_name]
            self.llm = OpenAI(
                openai_api_key=OPENAI_CONFIG["api_key"],
                model_name=model_name,
                temperature=temperature if temperature is not None else config["temperature"],
                max_tokens=config["max_tokens"]
            )
        else:
            if model_name is None:
                model_name = "llama-2-7b"
            self.llm = LocalLLM(
                model_name=model_name,
                temperature=temperature if temperature is not None else LLAMAFILE_CONFIG["models"][model_name]["temperature"]
            )

    def get_llm(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            str: Generated response
        """
        return self.llm.invoke(prompt)

    @property
    def backend_type(self) -> str:
        """Get the type of backend being used.
        
        Returns:
            str: Either 'openai' or 'llamafile'
        """
        return "openai" if self.use_openai else "llamafile"
