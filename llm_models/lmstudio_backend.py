"""LM Studio backend for local LLM inference."""
import os
from typing import Dict, Any, Optional, List
import yaml
import requests
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

class LMStudioLLM(BaseLLM):
    """LM Studio LLM wrapper."""
    
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "lmstudio"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate completions for multiple prompts."""
        headers = {
            "Content-Type": "application/json"
        }
        
        generations = []
        for prompt in prompts:
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stop": stop or self.stop
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            generations.append([Generation(text=text, generation_info={})])
        
        return LLMResult(generations=generations)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LM Studio API."""
        result = self._generate([prompt], stop=stop, run_manager=run_manager, **kwargs)
        return result.generations[0][0].text

class LMStudioEmbeddings:
    """LM Studio embeddings wrapper."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": texts
        }
        
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        return [item["embedding"] for item in response.json()["data"]]
    
    def embed_query(self, text: str) -> List[float]:
        """Get embeddings for a single text."""
        return self.embed_documents([text])[0]

def load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "llm_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["llm"]

def create_llm() -> LMStudioLLM:
    """Create LM Studio LLM instance."""
    config = load_llm_config()
    return LMStudioLLM(
        base_url=config["base_url"],
        model=config["chat_model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        frequency_penalty=config["frequency_penalty"],
        presence_penalty=config["presence_penalty"],
        stop=config["stop"]
    )

def create_embeddings() -> LMStudioEmbeddings:
    """Create LM Studio embeddings instance."""
    config = load_llm_config()
    return LMStudioEmbeddings(
        base_url=config["base_url"],
        model=config["embedding_model"]
    )