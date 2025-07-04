import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from utils.logging_util import LoggerMixin
from nn_models.agent_nn_v2 import TaskMetrics

@dataclass
class AgentScore:
    """Score and metadata for agent selection."""
    agent_name: str
    embedding_score: float
    nn_score: float
    combined_score: float
    historical_performance: Optional[float] = None

class MetaLearner(nn.Module, LoggerMixin):
    """Meta-learner for agent selection and performance optimization."""
    
    def __init__(self,
                 embedding_size: int = 768,
                 feature_size: int = 64,
                 hidden_size: int = 256):
        """Initialize meta-learner.
        
        Args:
            embedding_size: Size of task embeddings
            feature_size: Size of agent NN features
            hidden_size: Size of hidden layers
        """
        super().__init__()
        LoggerMixin.__init__(self)
        
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        
        # Network for combining embeddings and features
        self.network = nn.Sequential(
            nn.Linear(embedding_size + feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output is a score between 0 and 1
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # Historical performance tracking
        self.agent_metrics: Dict[str, List[TaskMetrics]] = {}
        self.training_history: List[Dict[str, float]] = []

    def forward(self,
                task_embedding: torch.Tensor,
                agent_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            task_embedding: Task embedding tensor
            agent_features: Agent NN features tensor
            
        Returns:
            torch.Tensor: Agent selection scores
        """
        # Concatenate embeddings and features
        combined = torch.cat([task_embedding, agent_features], dim=1)
        return self.network(combined)

    def train_step(self,
                  task_embedding: torch.Tensor,
                  agent_features: torch.Tensor,
                  success_score: torch.Tensor) -> float:
        """Perform one training step.
        
        Args:
            task_embedding: Task embedding tensor
            agent_features: Agent NN features tensor
            success_score: Target success score (0 to 1)
            
        Returns:
            float: Loss value
        """
        self.optimizer.zero_grad()
        predicted_score = self(task_embedding, agent_features)
        loss = self.criterion(predicted_score, success_score)
        loss.backward()
        self.optimizer.step()
        
        # Log training metrics
        metrics = {
            "meta_learner_loss": loss.item(),
            "predicted_score": predicted_score.mean().item(),
            "target_score": success_score.mean().item()
        }
        self.training_history.append(metrics)
        self.log_model_performance("meta_learner", metrics)
        
        return loss.item()

    def score_agents(self,
                    task_embedding: torch.Tensor,
                    agents_features: Dict[str, torch.Tensor]) -> List[AgentScore]:
        """Score agents for a given task.
        
        Args:
            task_embedding: Task embedding tensor
            agents_features: Dictionary of agent name to features tensor
            
        Returns:
            List[AgentScore]: Sorted list of agent scores
        """
        scores = []
        
        for agent_name, features in agents_features.items():
            # Calculate embedding similarity score
            embedding_score = self._calculate_embedding_similarity(
                task_embedding,
                features[:, :self.embedding_size]  # Use first part as embedding
            )
            
            # Get NN score
            with torch.no_grad():
                nn_score = self(task_embedding, features).item()
                
            # Get historical performance
            historical_score = self._get_historical_performance(agent_name)
            
            # Combine scores (weighted average)
            combined_score = self._combine_scores(
                embedding_score,
                nn_score,
                historical_score
            )
            
            scores.append(AgentScore(
                agent_name=agent_name,
                embedding_score=embedding_score,
                nn_score=nn_score,
                combined_score=combined_score,
                historical_performance=historical_score
            ))
        
        # Sort by combined score
        return sorted(scores, key=lambda x: x.combined_score, reverse=True)

    def update_metrics(self, agent_name: str, metrics: TaskMetrics):
        """Update historical metrics for an agent.
        
        Args:
            agent_name: Name of the agent
            metrics: Task execution metrics
        """
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = []
            
        self.agent_metrics[agent_name].append(metrics)
        
        # Log metrics
        self.log_event(
            "agent_metrics_update",
            {"agent": agent_name},
            metrics={
                "response_time": metrics.response_time,
                "confidence": metrics.confidence_score
            }
        )

    def _calculate_embedding_similarity(self,
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

    def _get_historical_performance(self, agent_name: str) -> Optional[float]:
        """Calculate historical performance score for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Optional[float]: Performance score or None if no history
        """
        if agent_name not in self.agent_metrics:
            return None
            
        recent_metrics = self.agent_metrics[agent_name][-100:]  # Last 100 tasks
        if not recent_metrics:
            return None
            
        # Calculate weighted score from metrics
        scores = []
        for metrics in recent_metrics:
            score = 0.0
            components = 0
            
            # Response time (lower is better)
            if metrics.response_time > 0:
                score += 1.0 / (1.0 + metrics.response_time)
                components += 1
                
            # Confidence score
            if metrics.confidence_score is not None:
                score += metrics.confidence_score
                components += 1
                
            # User feedback
            if metrics.user_feedback is not None:
                score += metrics.user_feedback / 5.0  # Normalize to [0,1]
                components += 1
                
            # Task success
            if metrics.task_success is not None:
                score += 1.0 if metrics.task_success else 0.0
                components += 1
                
            if components > 0:
                scores.append(score / components)
                
        return np.mean(scores) if scores else None

    def _combine_scores(self,
                       embedding_score: float,
                       nn_score: float,
                       historical_score: Optional[float]) -> float:
        """Combine different scores into final score.
        
        Args:
            embedding_score: Embedding similarity score
            nn_score: Neural network score
            historical_score: Historical performance score
            
        Returns:
            float: Combined score
        """
        # Weights for different components
        w_embedding = 0.3
        w_nn = 0.4
        w_historical = 0.3
        
        # Start with embedding and NN scores
        if historical_score is None:
            # Adjust weights if no historical data
            w_embedding = 0.4
            w_nn = 0.6
            combined = w_embedding * embedding_score + w_nn * nn_score
        else:
            combined = (w_embedding * embedding_score +
                      w_nn * nn_score +
                      w_historical * historical_score)
            
        return combined

    def save_model(self, path: str):
        """Save model state and metrics.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'agent_metrics': self.agent_metrics,
            'training_history': self.training_history
        }, path)
        
        self.log_event(
            "model_saved",
            {"path": path}
        )

    def load_model(self, path: str):
        """Load model state and metrics.
        
        Args:
            path: Path to load model from
        """
        try:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.agent_metrics = checkpoint['agent_metrics']
            self.training_history = checkpoint['training_history']
            
            self.log_event(
                "model_loaded",
                {"path": path}
            )
        except Exception as e:
            self.log_error(e, {"path": path})