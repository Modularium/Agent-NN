"""Neural network model for agent selection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

class AgentSelectorModel(nn.Module):
    """Neural network for selecting the best agent."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 num_agents: int,
                 num_metrics: int,
                 dropout: float = 0.1):
        """Initialize model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            num_agents: Number of agents
            num_metrics: Number of metrics to predict
            dropout: Dropout probability
        """
        super().__init__()
        
        # Input layer
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.agent_head = nn.Linear(prev_dim, num_agents)
        self.success_head = nn.Linear(prev_dim, 1)
        self.metrics_head = nn.Linear(prev_dim, num_metrics)
        
    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (agent_logits, success_prob, metrics)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Agent selection logits
        agent_logits = self.agent_head(features)
        
        # Success probability
        success_prob = torch.sigmoid(self.success_head(features))
        
        # Performance metrics
        metrics = self.metrics_head(features)
        
        return agent_logits, success_prob, metrics
        
    def predict_agent(self,
                     x: torch.Tensor,
                     threshold: float = 0.5) -> Tuple[int, float, torch.Tensor]:
        """Predict best agent for input.
        
        Args:
            x: Input tensor
            threshold: Success probability threshold
            
        Returns:
            Tuple of (agent_idx, success_prob, metrics)
        """
        with torch.no_grad():
            # Get predictions
            agent_logits, success_prob, metrics = self(x)
            
            # Get agent with highest probability
            agent_probs = F.softmax(agent_logits, dim=1)
            agent_idx = agent_probs.argmax(dim=1)
            
            # Check success probability
            if success_prob.item() < threshold:
                return None, success_prob.item(), metrics
                
            return agent_idx.item(), success_prob.item(), metrics

class AgentSelectorTrainer:
    """Trainer for agent selector model."""
    
    def __init__(self,
                 model: AgentSelectorModel,
                 optimizer: torch.optim.Optimizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer to use
            device: Device to use
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        # Loss functions
        self.agent_criterion = nn.CrossEntropyLoss()
        self.success_criterion = nn.BCELoss()
        self.metrics_criterion = nn.MSELoss()
        
    def train_step(self,
                  features: torch.Tensor,
                  agent_labels: torch.Tensor,
                  success_labels: torch.Tensor,
                  metric_labels: torch.Tensor) -> Dict[str, float]:
        """Perform one training step.
        
        Args:
            features: Input features
            agent_labels: Agent labels
            success_labels: Success labels
            metric_labels: Metric labels
            
        Returns:
            Dict containing losses
        """
        # Move data to device
        features = features.to(self.device)
        agent_labels = agent_labels.to(self.device)
        success_labels = success_labels.to(self.device)
        metric_labels = metric_labels.to(self.device)
        
        # Forward pass
        agent_logits, success_prob, metrics = self.model(features)
        
        # Calculate losses
        agent_loss = self.agent_criterion(agent_logits, agent_labels)
        success_loss = self.success_criterion(success_prob, success_labels)
        metrics_loss = self.metrics_criterion(metrics, metric_labels)
        
        # Combined loss
        total_loss = agent_loss + success_loss + metrics_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "agent_loss": agent_loss.item(),
            "success_loss": success_loss.item(),
            "metrics_loss": metrics_loss.item(),
            "total_loss": total_loss.item()
        }
        
    def validate_step(self,
                     features: torch.Tensor,
                     agent_labels: torch.Tensor,
                     success_labels: torch.Tensor,
                     metric_labels: torch.Tensor) -> Dict[str, float]:
        """Perform one validation step.
        
        Args:
            features: Input features
            agent_labels: Agent labels
            success_labels: Success labels
            metric_labels: Metric labels
            
        Returns:
            Dict containing metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            features = features.to(self.device)
            agent_labels = agent_labels.to(self.device)
            success_labels = success_labels.to(self.device)
            metric_labels = metric_labels.to(self.device)
            
            # Forward pass
            agent_logits, success_prob, metrics = self.model(features)
            
            # Calculate metrics
            agent_preds = agent_logits.argmax(dim=1)
            agent_acc = (agent_preds == agent_labels).float().mean()
            
            success_preds = (success_prob > 0.5).float()
            success_acc = (success_preds == success_labels).float().mean()
            
            metrics_mse = F.mse_loss(metrics, metric_labels)
            
        self.model.train()
        
        return {
            "agent_accuracy": agent_acc.item(),
            "success_accuracy": success_acc.item(),
            "metrics_mse": metrics_mse.item()
        }
        
    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Dict containing metrics
        """
        # Training metrics
        train_metrics = {
            "agent_loss": 0.0,
            "success_loss": 0.0,
            "metrics_loss": 0.0,
            "total_loss": 0.0
        }
        
        # Train steps
        for features, targets in train_loader:
            agent_labels = targets[:, 0].long()
            success_labels = targets[:, 1]
            metric_labels = targets[:, 2:]
            
            step_metrics = self.train_step(
                features,
                agent_labels,
                success_labels,
                metric_labels
            )
            
            for k, v in step_metrics.items():
                train_metrics[k] += v / len(train_loader)
                
        # Validation
        val_metrics = {}
        if val_loader is not None:
            val_metrics = {
                "agent_accuracy": 0.0,
                "success_accuracy": 0.0,
                "metrics_mse": 0.0
            }
            
            for features, targets in val_loader:
                agent_labels = targets[:, 0].long()
                success_labels = targets[:, 1]
                metric_labels = targets[:, 2:]
                
                step_metrics = self.validate_step(
                    features,
                    agent_labels,
                    success_labels,
                    metric_labels
                )
                
                for k, v in step_metrics.items():
                    val_metrics[k] += v / len(val_loader)
                    
        return {**train_metrics, **val_metrics}
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])