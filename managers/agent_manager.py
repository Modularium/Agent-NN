from typing import List, Dict, Optional
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agents.worker_agent import WorkerAgent
from config import LLM_API_KEY

class AgentManager:
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
        self.agents: Dict[str, WorkerAgent] = {}
        self.embeddings = OpenAIEmbeddings(openai_api_key=LLM_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize default agents with domain knowledge
        for domain, knowledge in self.DOMAIN_KNOWLEDGE.items():
            agent_name = f"{domain}_agent"
            self.agents[agent_name] = WorkerAgent(
                name=domain,
                domain_docs=[Document(page_content=text, metadata={"domain": domain}) 
                           for text in knowledge]
            )

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
            
        return {
            "name": agent_name,
            "domain": agent.name,
            "knowledge_base_size": len(agent.search_knowledge_base("", k=1000))
        }
