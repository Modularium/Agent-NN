from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from utils.agent_descriptions import (
    get_agent_description,
    get_agent_embedding_text,
    get_task_requirements,
    match_task_to_domain
)
from mlflow_integration.experiment_tracking import ExperimentTracker

class NNManager:
    def __init__(self):
        """Initialize the neural network manager for agent selection."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
        self.experiment_tracker = ExperimentTracker("agent_selection")
        
        # Cache for embeddings to avoid recomputing
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Thresholds and weights for agent selection
        self.confidence_threshold = 0.7
        self.weights = {
            "embedding_similarity": 0.6,
            "requirement_match": 0.4
        }
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []

    def predict_best_agent(self, task_description: str, available_agents: List[str]) -> Optional[str]:
        """Predict the best agent for a given task using multiple criteria.
        
        Args:
            task_description: Description of the task
            available_agents: List of available agent names
            
        Returns:
            str: Name of the best agent, or None if no suitable agent found
        """
        if not available_agents:
            return None
            
        # Get task embedding and requirements
        task_embedding = self._get_embedding(task_description)
        task_requirements = get_task_requirements(task_description)
        
        # Calculate scores for each agent
        agent_scores = self._compute_agent_scores(
            task_embedding=task_embedding,
            task_requirements=task_requirements,
            available_agents=available_agents
        )
        
        # Log the prediction event
        self.experiment_tracker.log_agent_selection(
            task_description=task_description,
            chosen_agent=max(agent_scores.items(), key=lambda x: x[1])[0],
            available_agents=available_agents,
            agent_scores=agent_scores,
            execution_result={"timestamp": datetime.now().isoformat()}
        )
        
        # Return best agent if score is above threshold
        best_agent, best_score = max(agent_scores.items(), key=lambda x: x[1])
        return best_agent if best_score >= self.confidence_threshold else None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = np.array(self.embeddings.embed_query(text))
        return self.embedding_cache[text]

    def _compute_agent_scores(self,
                            task_embedding: np.ndarray,
                            task_requirements: List[str],
                            available_agents: List[str]) -> Dict[str, float]:
        """Compute comprehensive scores for each agent.
        
        Args:
            task_embedding: Embedding of the task description
            task_requirements: List of task requirements
            available_agents: List of available agents
            
        Returns:
            Dict mapping agent names to scores
        """
        scores = {}
        for agent in available_agents:
            # Get base domain from agent name
            base_domain = agent.split('_')[0]
            
            # Get agent description and embedding
            agent_text = get_agent_embedding_text(base_domain)
            agent_embedding = self._get_embedding(agent_text)
            
            # Calculate embedding similarity score
            embedding_score = float(cosine_similarity(
                task_embedding.reshape(1, -1),
                agent_embedding.reshape(1, -1)
            )[0, 0])
            
            # Calculate requirement match score
            requirement_score = match_task_to_domain(task_requirements, base_domain)
            
            # Combine scores using weights
            scores[agent] = (
                self.weights["embedding_similarity"] * embedding_score +
                self.weights["requirement_match"] * requirement_score
            )
            
        return scores
    
    def update_model(self,
                    task_description: str,
                    chosen_agent: str,
                    execution_result: Dict[str, Any]):
        """Update the model based on task execution results.
        
        Args:
            task_description: Description of the executed task
            chosen_agent: Name of the agent that executed the task
            execution_result: Result of task execution including metrics
        """
        # Calculate success score
        success_score = float(execution_result.get("success", False))
        if success_score:
            # Adjust score based on execution time and other metrics
            execution_time = execution_result.get("execution_time", 0)
            if execution_time > 0:
                time_penalty = min(execution_time / 60.0, 0.5)  # Penalty for long execution
                success_score = max(0.5, success_score - time_penalty)
        
        # Update performance history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "chosen_agent": chosen_agent,
            "success_score": success_score,
            "execution_result": execution_result
        })
        
        # Calculate recent performance metrics
        recent_performance = self._calculate_recent_performance()
        
        # Log the update event
        self.experiment_tracker.log_model_update(
            model_type="agent_selection",
            parameters={
                "confidence_threshold": self.confidence_threshold,
                "weights": self.weights
            },
            metrics={
                "success_score": success_score,
                "recent_success_rate": recent_performance["success_rate"],
                "recent_avg_execution_time": recent_performance["avg_execution_time"]
            }
        )
        
        # Update model parameters based on performance
        self._update_parameters(recent_performance)
    
    def _calculate_recent_performance(self, window_size: int = 50) -> Dict[str, float]:
        """Calculate performance metrics over recent tasks.
        
        Args:
            window_size: Number of recent tasks to consider
            
        Returns:
            Dict containing performance metrics
        """
        recent_tasks = self.performance_history[-window_size:]
        if not recent_tasks:
            return {"success_rate": 0.0, "avg_execution_time": 0.0}
            
        success_rate = sum(1 for task in recent_tasks 
                         if task["success_score"] > 0.5) / len(recent_tasks)
        
        execution_times = [task["execution_result"].get("execution_time", 0) 
                         for task in recent_tasks]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time
        }
    
    def _update_parameters(self, performance_metrics: Dict[str, float]):
        """Update model parameters based on performance metrics.
        
        Args:
            performance_metrics: Dict containing performance metrics
        """
        # Adjust confidence threshold based on success rate
        if performance_metrics["success_rate"] > 0.8:
            # High success rate - we can be more lenient
            self.confidence_threshold = max(0.6, self.confidence_threshold - 0.02)
        elif performance_metrics["success_rate"] < 0.5:
            # Low success rate - be more strict
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.02)
        
        # Adjust weights based on execution time
        if performance_metrics["avg_execution_time"] > 30:  # If avg time > 30 seconds
            # Increase weight of requirement matching to be more selective
            self.weights["requirement_match"] = min(0.6, self.weights["requirement_match"] + 0.05)
            self.weights["embedding_similarity"] = 1 - self.weights["requirement_match"]
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics and model parameters.
        
        Returns:
            Dict containing metrics and parameters
        """
        recent_performance = self._calculate_recent_performance()
        return {
            "model_parameters": {
                "confidence_threshold": self.confidence_threshold,
                "weights": self.weights
            },
            "performance_metrics": recent_performance,
            "total_tasks_processed": len(self.performance_history),
            "embedding_cache_size": len(self.embedding_cache)
        }
