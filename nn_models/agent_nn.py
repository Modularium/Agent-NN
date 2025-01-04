import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class TaskMetrics:
    """Metrics for task execution performance."""
    response_time: float  # Time taken to respond
    confidence_score: float  # Model's confidence in response
    user_feedback: Optional[float] = None  # User rating (if available)
    task_success: Optional[bool] = None  # Whether task was completed successfully

class AgentNN(nn.Module):
    """Neural network for optimizing WorkerAgent task performance."""
    
    def __init__(self, input_size: int = 768, hidden_size: int = 256, output_size: int = 64):
        """Initialize the neural network.
        
        Args:
            input_size: Size of input embeddings (default: 768 for most transformers)
            hidden_size: Size of hidden layer
            output_size: Size of output layer (task-specific features)
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()  # Output between -1 and 1
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Track training history
        self.training_losses: List[float] = []
        self.eval_metrics: List[Dict[str, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def train_step(self, task_embedding: torch.Tensor, target_features: torch.Tensor) -> float:
        """Perform one training step.
        
        Args:
            task_embedding: Embedding of the task description
            target_features: Target task-specific features
            
        Returns:
            float: Loss value
        """
        self.optimizer.zero_grad()
        output = self(task_embedding)
        loss = self.criterion(output, target_features)
        loss.backward()
        self.optimizer.step()
        
        self.training_losses.append(loss.item())
        return loss.item()

    def predict_task_features(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """Predict task-specific features from task embedding.
        
        Args:
            task_embedding: Embedding of the task description
            
        Returns:
            torch.Tensor: Predicted task-specific features
        """
        with torch.no_grad():
            return self(task_embedding)

    def evaluate_performance(self, task_metrics: TaskMetrics) -> Dict[str, float]:
        """Evaluate agent's performance on a task.
        
        Args:
            task_metrics: Metrics from task execution
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        metrics = {
            "response_time": task_metrics.response_time,
            "confidence": task_metrics.confidence_score
        }
        
        if task_metrics.user_feedback is not None:
            metrics["user_feedback"] = task_metrics.user_feedback
        
        if task_metrics.task_success is not None:
            metrics["success_rate"] = 1.0 if task_metrics.task_success else 0.0
        
        self.eval_metrics.append(metrics)
        return metrics

    def get_training_summary(self) -> Dict[str, float]:
        """Get summary of training progress.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        if not self.training_losses:
            return {}
            
        return {
            "avg_loss": np.mean(self.training_losses[-100:]),  # Last 100 batches
            "min_loss": min(self.training_losses),
            "max_loss": max(self.training_losses),
            "total_batches": len(self.training_losses)
        }

    def save_model(self, path: str) -> None:
        """Save model state to disk.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'eval_metrics': self.eval_metrics
        }, path)

    def load_model(self, path: str) -> None:
        """Load model state from disk.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint['training_losses']
        self.eval_metrics = checkpoint['eval_metrics']