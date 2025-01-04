from typing import Optional, Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from .llm_backend import LLMBackendManager, LLMBackendType

class SpecializedLLM:
    # Default domain configurations
    DOMAIN_CONFIGS = {
        "finance": {
            "temperature": 0.1,
            "system_prompt": """You are a financial expert assistant. Provide accurate, factual information about financial topics.
            Always cite sources when possible and maintain professional financial terminology.
            If you're unsure about any financial information, clearly state your uncertainty.""",
            "preferred_models": {
                "openai": "gpt-4",  # More precise for financial calculations
                "lmstudio": "local-model",
                "llamafile": "llama-2-7b"
            }
        },
        "tech": {
            "temperature": 0.3,
            "system_prompt": """You are a technical expert assistant. Provide clear, accurate technical explanations.
            Use precise technical terminology and include code examples when relevant.
            Always consider best practices and security implications in your responses.""",
            "preferred_models": {
                "openai": "gpt-3.5-turbo",
                "lmstudio": "local-model",
                "llamafile": "llama-2-7b"
            }
        },
        "marketing": {
            "temperature": 0.7,
            "system_prompt": """You are a marketing expert assistant. Help create engaging and effective marketing content.
            Consider target audience, brand voice, and marketing objectives in your responses.
            Provide creative suggestions while maintaining marketing best practices.""",
            "preferred_models": {
                "openai": "gpt-3.5-turbo",
                "lmstudio": "local-model",
                "llamafile": "llama-2-7b"
            }
        }
    }

    def __init__(self, domain: str, backend_type: Optional[LLMBackendType] = None):
        """Initialize a specialized LLM for a specific domain.
        
        Args:
            domain: The domain specialization (e.g., "finance", "tech", "marketing")
            backend_type: Optional backend type to use (OpenAI, LM Studio, or Llamafile)
        """
        self.domain = domain.lower()
        
        # Get domain-specific configuration or use defaults
        self.config = self.DOMAIN_CONFIGS.get(self.domain, {
            "temperature": 0.2,
            "system_prompt": f"You are an expert assistant in {domain}. Provide accurate and helpful information.",
            "preferred_models": {
                "openai": "gpt-3.5-turbo",
                "lmstudio": "local-model",
                "llamafile": "llama-2-7b"
            }
        })
        
        # Initialize LLM backend manager
        self.backend_manager = LLMBackendManager()
        if backend_type:
            self.backend_manager.set_backend(backend_type)
        
        # Get appropriate LLM for the domain and backend
        self.llm = self._get_domain_llm()
        
        # Create domain-specific prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=f"{self.config['system_prompt']}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:"
        )
        
        # Create RunnableSequence with the specialized prompt
        self.chain = RunnablePassthrough() | self.prompt_template | self.llm

    def _get_domain_llm(self):
        """Get the appropriate LLM for the domain."""
        backend_type = self.backend_manager.current_backend
        model_name = self.config["preferred_models"][backend_type.value]
        
        return self.backend_manager.get_llm(
            backend_type=backend_type,
            model_name=model_name,
            temperature=self.config["temperature"]
        )

    def switch_backend(self, backend_type: LLMBackendType):
        """Switch to a different LLM backend.
        
        Args:
            backend_type: The backend type to switch to
        """
        self.backend_manager.set_backend(backend_type)
        self.llm = self._get_domain_llm()
        self.chain = RunnablePassthrough() | self.prompt_template | self.llm

    def get_llm(self):
        """Get the current LLM instance."""
        return self.llm
    
    def get_chain(self):
        """Get the specialized LLMChain with domain-specific prompting."""
        return self.chain
    
    def generate_response(self,
                         question: str,
                         context: Optional[str] = None,
                         backend_type: Optional[LLMBackendType] = None) -> str:
        """Generate a domain-specific response to a question.
        
        Args:
            question: The question to answer
            context: Optional context to include in the prompt
            backend_type: Optional backend to use for this specific response
            
        Returns:
            str: The generated response
        """
        if backend_type and backend_type != self.backend_manager.current_backend:
            # Temporarily switch backend for this response
            original_backend = self.backend_manager.current_backend
            self.switch_backend(backend_type)
            try:
                response = self.chain.invoke({
                    "question": question,
                    "context": context or "No additional context provided."
                }).content
            finally:
                # Switch back to original backend
                self.switch_backend(original_backend)
            return response
            
        return self.chain.invoke({
            "question": question,
            "context": context or "No additional context provided."
        }).content
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend configuration.
        
        Returns:
            Dict containing backend information
        """
        return {
            "current_backend": self.backend_manager.current_backend.value,
            "current_model": self.config["preferred_models"][self.backend_manager.current_backend.value],
            "available_models": self.backend_manager.get_available_models(),
            "domain": self.domain,
            "temperature": self.config["temperature"]
        }
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the current backend.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        embeddings = self.backend_manager.get_embeddings()
        return embeddings.embed_query(text)
