from typing import List, Dict, Optional, Tuple
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
import mlflow
from config import LLM_API_KEY, MLFLOW_TRACKING_URI

class NNManager:
    def __init__(self):
        """Initialize the neural network manager for agent selection."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=LLM_API_KEY)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Cache for embeddings to avoid recomputing
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Threshold for agent selection confidence
        self.confidence_threshold = 0.7

    def predict_best_agent(self, task_description: str, available_agents: List[str]) -> Optional[str]:
        """Predict the best agent for a given task.
        
        Args:
            task_description: Description of the task
            available_agents: List of available agent names
            
        Returns:
            str: Name of the best agent, or None if no suitable agent found
        """
        if not available_agents:
            return None
            
        # Start MLflow run for tracking
        with mlflow.start_run(run_name="agent_selection") as run:
            # Log task description
            mlflow.log_param("task_description", task_description)
            mlflow.log_param("available_agents", available_agents)
            
            # Get task embedding
            task_embedding = self._get_embedding(task_description)
            
            # Get agent scores
            agent_scores = self._compute_agent_scores(task_embedding, available_agents)
            
            # Log scores
            for agent, score in agent_scores.items():
                mlflow.log_metric(f"score_{agent}", score)
            
            # Get best agent if score is above threshold
            best_agent, best_score = max(agent_scores.items(), key=lambda x: x[1])
            
            if best_score >= self.confidence_threshold:
                mlflow.log_metric("selection_confidence", best_score)
                mlflow.log_param("selected_agent", best_agent)
                return best_agent
            
            mlflow.log_param("selected_agent", "None")
            return None

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.embeddings.embed_query(text)
        return self.embedding_cache[text]

    def _compute_agent_scores(self, task_embedding: List[float], agents: List[str]) -> Dict[str, float]:
        """Compute similarity scores between task and each agent.
        
        Args:
            task_embedding: Embedding of the task description
            agents: List of agent names
            
        Returns:
            Dict mapping agent names to similarity scores
        """
        # Get agent descriptions
        agent_descriptions = {
            "finance_agent": """Financial expert agent specializing in:
                - Financial analysis and reporting
                - Budgeting and forecasting
                - Investment strategies
                - Risk management""",
            "tech_agent": """Technical expert agent specializing in:
                - Software development
                - System architecture
                - Technical problem-solving
                - DevOps practices""",
            "marketing_agent": """Marketing expert agent specializing in:
                - Marketing strategy
                - Digital marketing
                - Brand development
                - Customer engagement"""
        }
        
        scores = {}
        for agent in agents:
            # Get base domain from agent name (remove _agent suffix and numbers)
            base_domain = agent.split('_')[0]
            
            # Get description for the agent's domain
            description = agent_descriptions.get(
                f"{base_domain}_agent",
                f"Expert agent specializing in {base_domain} related tasks."
            )
            
            # Compute similarity score
            agent_embedding = self._get_embedding(description)
            score = self._compute_similarity(task_embedding, agent_embedding)
            scores[agent] = score
            
        return scores

    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def update_model(self, task_description: str, chosen_agent: str, success_score: float):
        """Update the model based on task execution results.
        
        Args:
            task_description: Description of the executed task
            chosen_agent: Name of the agent that executed the task
            success_score: Score indicating how well the task was executed (0-1)
        """
        with mlflow.start_run(run_name="model_update") as run:
            mlflow.log_param("task_description", task_description)
            mlflow.log_param("chosen_agent", chosen_agent)
            mlflow.log_metric("success_score", success_score)
            
            # In a more advanced implementation, this would update the model weights
            # For now, we just adjust the confidence threshold based on success
            if success_score > 0.8:
                self.confidence_threshold = max(0.6, self.confidence_threshold - 0.01)
            elif success_score < 0.5:
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.01)
            
            mlflow.log_metric("new_confidence_threshold", self.confidence_threshold)
