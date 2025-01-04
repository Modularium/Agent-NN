from typing import List, Dict, Optional, Tuple
import torch
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agents.worker_agent import WorkerAgent
from managers.hybrid_matcher import HybridMatcher, MatchResult
from nn_models.agent_nn import TaskMetrics
from utils.logging_util import LoggerMixin
from config import LLM_BACKEND
from config.llm_config import OPENAI_CONFIG, LMSTUDIO_CONFIG

class AgentManager(LoggerMixin):
    # Default domain knowledge for initial agents
    DOMAIN_KNOWLEDGE = {
        "finance": [
            "Financial analysis involves examining financial statements, cash flows, and market trends.",
            "Key financial metrics include ROI, ROE, P/E ratio, and debt-to-equity ratio.",
            "Risk management in finance covers market risk, credit risk, and operational risk.",
            "Financial planning includes budgeting, investment strategy, and retirement planning."
        ],
        "tech": [
            "Software development best practices include version control, testing, and documentation.",
            "Common programming paradigms: object-oriented, functional, and procedural programming.",
            "System architecture patterns: microservices, monolithic, and serverless architectures.",
            "DevOps practices include continuous integration, deployment, and infrastructure as code."
        ],
        "marketing": [
            "Marketing strategy encompasses market research, targeting, positioning, and branding.",
            "Digital marketing channels include SEO, social media, email, and content marketing.",
            "Customer segmentation helps target specific audience groups effectively.",
            "Marketing analytics measures KPIs like conversion rate, CAC, and customer lifetime value."
        ]
    }

    def __init__(self):
        """Initialize the agent manager with default agents."""
        super().__init__()
        self.agents: Dict[str, WorkerAgent] = {}
        
        # Initialize embeddings based on backend
        if LLM_BACKEND == "openai":
            self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_CONFIG["api_key"])
        else:  # Default to LM-Studio
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize hybrid matcher
        self.matcher = HybridMatcher()
        
        # Cache for embeddings and features
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        self.feature_cache: Dict[str, torch.Tensor] = {}
        
        # Initialize default agents with domain knowledge
        for domain, knowledge in self.DOMAIN_KNOWLEDGE.items():
            agent_name = f"{domain}_agent"
            self.agents[agent_name] = WorkerAgent(
                name=domain,
                domain_docs=[Document(page_content=text, metadata={"domain": domain}) 
                           for text in knowledge]
            )
            
            # Cache agent embeddings and features
            self._update_agent_cache(agent_name)

    def get_all_agents(self) -> List[str]:
        """Get list of all available agent names."""
        return list(self.agents.keys())

    def get_agent(self, name: str) -> Optional[WorkerAgent]:
        """Get a specific agent by name."""
        return self.agents.get(name)

    def create_new_agent(self, task_description: str) -> WorkerAgent:
        """Create a new agent based on task description.
        
        Args:
            task_description: Description of the task to handle
            
        Returns:
            WorkerAgent: Newly created agent
        """
        # Infer domain and get relevant knowledge
        domain = self._infer_domain(task_description)
        domain_knowledge = self.DOMAIN_KNOWLEDGE.get(domain, [
            "General knowledge base for handling various tasks.",
            "Problem-solving approaches and methodologies.",
            "Best practices for task execution and documentation."
        ])
        
        # Create unique name for the new agent
        new_agent_name = f"{domain}_agent_{len(self.agents)+1}"
        
        # Create agent with domain knowledge
        new_agent = WorkerAgent(
            name=domain,
            domain_docs=[Document(page_content=text, metadata={
                "domain": domain,
                "source": "initial_knowledge",
                "task_context": task_description
            }) for text in domain_knowledge]
        )
        
        self.agents[new_agent_name] = new_agent
        return new_agent

    def _infer_domain(self, task_description: str) -> str:
        """Infer the domain of a task using embeddings similarity.
        
        Args:
            task_description: Description of the task
            
        Returns:
            str: Inferred domain name
        """
        # Get embedding for task description
        task_embedding = self.embeddings.embed_query(task_description)
        
        # Get embeddings for domain descriptions
        domain_descriptions = {
            "finance": "Financial analysis, accounting, budgeting, investment, and risk management",
            "tech": "Software development, system architecture, programming, and technical solutions",
            "marketing": "Marketing strategy, digital marketing, branding, and customer engagement"
        }
        
        domain_embeddings = {
            domain: self.embeddings.embed_query(desc)
            for domain, desc in domain_descriptions.items()
        }
        
        # Calculate cosine similarity with each domain
        from numpy import dot
        from numpy.linalg import norm
        
        similarities = {
            domain: dot(task_embedding, domain_emb)/(norm(task_embedding)*norm(domain_emb))
            for domain, domain_emb in domain_embeddings.items()
        }
        
        # Return domain with highest similarity
        return max(similarities.items(), key=lambda x: x[1])[0]
    
    def get_agent_metadata(self, agent_name: str) -> Dict:
        """Get metadata about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dict containing agent metadata
        """
        agent = self.get_agent(agent_name)
        if not agent:
            return {}
            
        metadata = {
            "name": agent_name,
            "domain": agent.name,
            "knowledge_base_size": len(agent.search_knowledge_base("", k=1000))
        }
        
        # Add embeddings and features if available
        if agent_name in self.embedding_cache:
            metadata["embedding"] = self.embedding_cache[agent_name].tolist()
        if agent_name in self.feature_cache:
            metadata["features"] = self.feature_cache[agent_name].tolist()
            
        return metadata
        
    def select_agent(self, task_description: str) -> Tuple[WorkerAgent, MatchResult]:
        """Select the best agent for a task using hybrid matching.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Tuple[WorkerAgent, MatchResult]: Selected agent and match details
        """
        # Get task embedding
        task_embedding = torch.tensor(
            self.embeddings.embed_query(task_description)
        ).unsqueeze(0)
        
        # Prepare agent data for matcher
        agent_data = {
            name: {
                "embedding": self.embedding_cache[name],
                "features": self.feature_cache[name]
            }
            for name in self.agents.keys()
            if name in self.embedding_cache and name in self.feature_cache
        }
        
        # Get matches from hybrid matcher
        matches = self.matcher.match_task(
            task_description,
            task_embedding,
            agent_data
        )
        
        # Log matching results
        self.log_event(
            "agent_selection",
            {
                "task": task_description,
                "selected_agent": matches[0].agent_name,
                "confidence": matches[0].confidence,
                "match_details": matches[0].match_details
            }
        )
        
        # Return best match
        best_match = matches[0]
        return self.agents[best_match.agent_name], best_match
        
    def update_agent_performance(self,
                               agent_name: str,
                               task_metrics: TaskMetrics,
                               success_score: float):
        """Update agent performance metrics.
        
        Args:
            agent_name: Name of the agent
            task_metrics: Task execution metrics
            success_score: Task success score
        """
        self.matcher.update_agent_performance(
            agent_name,
            task_metrics,
            success_score
        )
        
        # Update cache if needed
        self._update_agent_cache(agent_name)
        
    def _update_agent_cache(self, agent_name: str):
        """Update cached embeddings and features for an agent.
        
        Args:
            agent_name: Name of the agent to update
        """
        agent = self.agents[agent_name]
        
        try:
            # Get agent description
            description = f"Domain: {agent.name}\n"
            description += "\n".join(
                doc.page_content
                for doc in agent.search_knowledge_base("", k=5)
            )
            
            # Update embedding cache
            self.embedding_cache[agent_name] = torch.tensor(
                self.embeddings.embed_query(description)
            ).unsqueeze(0)
            
            # Update feature cache
            self.feature_cache[agent_name] = agent.get_features()
            
            self.log_event(
                "cache_update",
                {
                    "agent": agent_name,
                    "cache_type": "both",
                    "status": "success"
                }
            )
            
        except Exception as e:
            self.log_error(e, {
                "agent": agent_name,
                "operation": "cache_update"
            })
            
    def save_state(self, path: str):
        """Save manager state.
        
        Args:
            path: Path to save state
        """
        self.matcher.save_state(f"{path}_matcher")
        self.log_event("state_saved", {"path": path})
        
    def load_state(self, path: str):
        """Load manager state.
        
        Args:
            path: Path to load state from
        """
        try:
            self.matcher.load_state(f"{path}_matcher")
            self.log_event("state_loaded", {"path": path})
        except Exception as e:
            self.log_error(e, {"path": path})
