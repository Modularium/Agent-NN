from typing import Optional, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .base_llm import BaseLLM

class SpecializedLLM:
    # Default domain configurations
    DOMAIN_CONFIGS = {
        "finance": {
            "temperature": 0.1,
            "system_prompt": """You are a financial expert assistant. Provide accurate, factual information about financial topics.
            Always cite sources when possible and maintain professional financial terminology.
            If you're unsure about any financial information, clearly state your uncertainty."""
        },
        "tech": {
            "temperature": 0.3,
            "system_prompt": """You are a technical expert assistant. Provide clear, accurate technical explanations.
            Use precise technical terminology and include code examples when relevant.
            Always consider best practices and security implications in your responses."""
        },
        "marketing": {
            "temperature": 0.7,
            "system_prompt": """You are a marketing expert assistant. Help create engaging and effective marketing content.
            Consider target audience, brand voice, and marketing objectives in your responses.
            Provide creative suggestions while maintaining marketing best practices."""
        }
    }

    def __init__(self, domain: str):
        """Initialize a specialized LLM for a specific domain.
        
        Args:
            domain: The domain specialization (e.g., "finance", "tech", "marketing")
        """
        self.domain = domain.lower()
        
        # Get domain-specific configuration or use defaults
        config = self.DOMAIN_CONFIGS.get(self.domain, {
            "temperature": 0.2,
            "system_prompt": f"You are an expert assistant in {domain}. Provide accurate and helpful information."
        })
        
        # Initialize base LLM with domain-specific temperature
        self.base_llm = BaseLLM(temperature=config["temperature"])
        
        # Create domain-specific prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=f"{config['system_prompt']}\n\nContext: {{context}}\n\nQuestion: {{question}}\n\nAnswer:"
        )
        
        # Create LLMChain with the specialized prompt
        self.chain = LLMChain(
            llm=self.base_llm.get_llm(),
            prompt=self.prompt_template
        )

    def get_llm(self):
        """Get the base LLM instance."""
        return self.base_llm.get_llm()
    
    def get_chain(self):
        """Get the specialized LLMChain with domain-specific prompting."""
        return self.chain
    
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """Generate a domain-specific response to a question.
        
        Args:
            question: The question to answer
            context: Optional context to include in the prompt
            
        Returns:
            str: The generated response
        """
        return self.chain.run({
            "question": question,
            "context": context or "No additional context provided."
        })
