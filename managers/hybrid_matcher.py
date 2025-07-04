from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass
from utils.logging_util import LoggerMixin
from managers.meta_learner import MetaLearner, AgentScore
from nn_models.agent_nn_v2 import TaskMetrics

@dataclass
class MatchResult:
    """Result of hybrid matching."""
    agent_name: str
    similarity_score: float
    nn_score: float
    combined_score: float
    confidence: float
    match_details: Dict[str, float]

class HybridMatcher(LoggerMixin):
    """Hybrid matching system combining embedding similarity and neural features."""
    
    def __init__(self,
                 embedding_size: int = 768,
                 feature_size: int = 64,
                 similarity_threshold: float = 0.7):
        """Initialize hybrid matcher.
        
        Args:
            embedding_size: Size of embeddings
            feature_size: Size of neural features
            similarity_threshold: Minimum similarity for agent match
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize meta-learner
        self.meta_learner = MetaLearner(
            embedding_size=embedding_size,
            feature_size=feature_size
        )
        
        # Cache for embeddings and features
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        self.feature_cache: Dict[str, torch.Tensor] = {}
        
        self.log_event(
            "matcher_initialized",
            {
                "embedding_size": embedding_size,
                "feature_size": feature_size,
                "similarity_threshold": similarity_threshold
            }
        )

    def match_task(self,
                   task_description: str,
                   task_embedding: torch.Tensor,
                   available_agents: Dict[str, Dict[str, torch.Tensor]]) -> List[MatchResult]:
        """Match task to available agents using hybrid approach.
        
        Args:
            task_description: Description of the task
            task_embedding: Task embedding tensor
            available_agents: Dict of agent name to their embeddings and features
            
        Returns:
            List[MatchResult]: Sorted list of match results
        """
        results = []
        
        # Get scores from meta-learner
        agent_scores = self.meta_learner.score_agents(
            task_embedding,
            {name: data["features"] for name, data in available_agents.items()}
        )
        
        # Calculate similarity scores
        for agent_name, data in available_agents.items():
            # Get agent embedding and features
            agent_embedding = data["embedding"]
            agent_features = data["features"]
            
            # Calculate similarity score
            similarity = self._calculate_similarity(task_embedding, agent_embedding)
            
            # Get NN score from meta-learner results
            nn_score = next(
                (score.nn_score for score in agent_scores
                 if score.agent_name == agent_name),
                0.0
            )
            
            # Calculate combined score
            combined_score = self._combine_scores(
                similarity,
                nn_score,
                agent_name
            )
            
            # Calculate confidence based on score distribution
            confidence = self._calculate_confidence(
                similarity,
                nn_score,
                combined_score
            )
            
            results.append(MatchResult(
                agent_name=agent_name,
                similarity_score=similarity,
                nn_score=nn_score,
                combined_score=combined_score,
                confidence=confidence,
                match_details={
                    "embedding_similarity": similarity,
                    "nn_score": nn_score,
                    "historical_performance": self._get_historical_performance(agent_name)
                }
            ))
            
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Log matching results
        self.log_event(
            "task_matching",
            {
                "task": task_description,
                "results": [
                    {
                        "agent": r.agent_name,
                        "score": r.combined_score,
                        "confidence": r.confidence
                    }
                    for r in results[:3]  # Log top 3
                ]
            }
        )
        
        return results

    def update_agent_performance(self,
                               agent_name: str,
                               task_metrics: TaskMetrics,
                               success_score: float):
        """Update agent performance metrics.
        
        Args:
            agent_name: Name of the agent
            task_metrics: Task execution metrics
            success_score: Task success score (0 to 1)
        """
        # Update meta-learner metrics
        self.meta_learner.update_metrics(agent_name, task_metrics)
        
        # Log performance update
        self.log_event(
            "agent_performance_update",
            {
                "agent": agent_name,
                "success_score": success_score
            },
            metrics={
                "response_time": task_metrics.response_time,
                "confidence": task_metrics.confidence_score,
                "success_score": success_score
            }
        )

    def train_meta_learner(self,
                          task_embeddings: torch.Tensor,
                          agent_features: torch.Tensor,
                          success_scores: torch.Tensor,
                          num_epochs: int = 10):
        """Train meta-learner on historical data.
        
        Args:
            task_embeddings: Task embedding tensors
            agent_features: Agent feature tensors
            success_scores: Success score tensors
            num_epochs: Number of training epochs
        """
        total_loss = 0.0
        for epoch in range(num_epochs):
            loss = self.meta_learner.train_step(
                task_embeddings,
                agent_features,
                success_scores
            )
            total_loss += loss
            
            # Log training progress
            self.log_event(
                "meta_learner_training",
                {
                    "epoch": epoch + 1,
                    "total_epochs": num_epochs
                },
                metrics={"loss": loss}
            )
            
        avg_loss = total_loss / num_epochs
        self.log_event(
            "meta_learner_training_complete",
            {"num_epochs": num_epochs},
            metrics={"avg_loss": avg_loss}
        )

    def _calculate_similarity(self,
                            task_embedding: torch.Tensor,
                            agent_embedding: torch.Tensor) -> float:
        """Calculate cosine similarity between embeddings.
        
        Args:
            task_embedding: Task embedding tensor
            agent_embedding: Agent embedding tensor
            
        Returns:
            float: Similarity score
        """
        return torch.cosine_similarity(
            task_embedding,
            agent_embedding,
            dim=1
        ).item()

    def _combine_scores(self,
                       similarity: float,
                       nn_score: float,
                       agent_name: str) -> float:
        """Combine similarity and NN scores.
        
        Args:
            similarity: Embedding similarity score
            nn_score: Neural network score
            agent_name: Name of the agent
            
        Returns:
            float: Combined score
        """
        # Get historical performance
        historical = self._get_historical_performance(agent_name)
        
        # Weights for different components
        w_similarity = 0.3
        w_nn = 0.4
        w_historical = 0.3 if historical is not None else 0.0
        
        # Adjust weights if no historical data
        if historical is None:
            w_similarity = 0.4
            w_nn = 0.6
            
        # Calculate combined score
        score = w_similarity * similarity + w_nn * nn_score
        if historical is not None:
            score += w_historical * historical
            
        return score

    def _calculate_confidence(self,
                            similarity: float,
                            nn_score: float,
                            combined_score: float) -> float:
        """Calculate confidence in match.
        
        Args:
            similarity: Embedding similarity score
            nn_score: Neural network score
            combined_score: Combined score
            
        Returns:
            float: Confidence score
        """
        # Base confidence on score consistency
        score_std = np.std([similarity, nn_score, combined_score])
        consistency = 1.0 - min(score_std, 0.5) / 0.5  # Normalize to [0,1]
        
        # Factor in absolute scores
        score_confidence = (similarity + nn_score + combined_score) / 3
        
        # Combine factors
        confidence = 0.6 * consistency + 0.4 * score_confidence
        
        return confidence

    def _get_historical_performance(self, agent_name: str) -> Optional[float]:
        """Get historical performance score for agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Optional[float]: Performance score or None if no history
        """
        return self.meta_learner._get_historical_performance(agent_name)

    def save_state(self, path: str):
        """Save matcher state.
        
        Args:
            path: Path to save state
        """
        self.meta_learner.save_model(path)
        self.log_event("state_saved", {"path": path})

    def load_state(self, path: str):
        """Load matcher state.
        
        Args:
            path: Path to load state from
        """
        try:
            self.meta_learner.load_model(path)
            self.log_event("state_loaded", {"path": path})
        except Exception as e:
            self.log_error(e, {"path": path})