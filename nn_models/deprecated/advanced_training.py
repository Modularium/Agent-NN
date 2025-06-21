# deprecated – moved for cleanup in v1.0.0-beta
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datetime import datetime
import mlflow
from utils.logging_util import LoggerMixin

class MultiModalDataset(Dataset):
    """Dataset for multi-modal agent training data."""
    
    def __init__(self,
                 text_embeddings: torch.Tensor,
                 performance_metrics: torch.Tensor,
                 feedback_scores: torch.Tensor):
        """Initialize dataset.
        
        Args:
            text_embeddings: Text embedding tensors
            performance_metrics: Performance metric tensors
            feedback_scores: Feedback score tensors
        """
        self.text_embeddings = text_embeddings
        self.performance_metrics = performance_metrics
        self.feedback_scores = feedback_scores
        
    def __len__(self) -> int:
        return len(self.text_embeddings)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.text_embeddings[idx],
            self.performance_metrics[idx],
            self.feedback_scores[idx]
        )

class HierarchicalNetwork(nn.Module):
    """Hierarchical neural network for agent task processing."""
    
    def __init__(self,
                 text_dim: int,
                 metric_dim: int,
                 feedback_dim: int,
                 hidden_dims: List[int],
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize network.
        
        Args:
            text_dim: Text embedding dimension
            metric_dim: Performance metric dimension
            feedback_dim: Feedback score dimension
            hidden_dims: Hidden layer dimensions
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Feature extraction layers
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        self.metric_encoder = nn.Sequential(
            nn.Linear(metric_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dims[1],
            num_heads,
            dropout=dropout
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dims[1] * 3, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], hidden_dims[3])
        )
        
        # Task-specific heads
        self.agent_selector = nn.Linear(hidden_dims[3], 1)
        self.performance_predictor = nn.Linear(hidden_dims[3], metric_dim)
        
    def forward(self,
                text_emb: torch.Tensor,
                metrics: torch.Tensor,
                feedback: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            text_emb: Text embeddings
            metrics: Performance metrics
            feedback: Feedback scores
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Agent selection and performance prediction
        """
        # Encode features
        text_features = self.text_encoder(text_emb)
        metric_features = self.metric_encoder(metrics)
        feedback_features = self.feedback_encoder(feedback)
        
        # Apply attention
        features = torch.stack([
            text_features,
            metric_features,
            feedback_features
        ], dim=0)
        
        attended_features, _ = self.attention(
            features,
            features,
            features
        )
        
        # Fuse features
        fused = self.fusion(
            torch.cat([
                attended_features[0],
                attended_features[1],
                attended_features[2]
            ], dim=-1)
        )
        
        # Task-specific predictions
        agent_scores = self.agent_selector(fused)
        performance_pred = self.performance_predictor(fused)
        
        return agent_scores, performance_pred

class AdvancedTrainer(LoggerMixin):
    """Advanced neural network trainer."""
    
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 gradient_clip: float = 1.0):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            learning_rate: Learning rate
            weight_decay: Weight decay
            gradient_clip: Gradient clipping value
        """
        super().__init__()
        self.model = model
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.gradient_clip = gradient_clip
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("advanced_training")
        
    def train_epoch(self,
                   dataloader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        agent_loss = 0.0
        perf_loss = 0.0
        
        for batch_idx, (text, metrics, feedback) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            agent_scores, perf_pred = self.model(text, metrics, feedback)
            
            # Calculate losses
            agent_criterion = nn.BCEWithLogitsLoss()
            perf_criterion = nn.MSELoss()
            
            batch_agent_loss = agent_criterion(
                agent_scores,
                torch.ones_like(agent_scores)  # Target scores
            )
            
            batch_perf_loss = perf_criterion(
                perf_pred,
                metrics  # Target metrics
            )
            
            # Combined loss
            loss = batch_agent_loss + batch_perf_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            agent_loss += batch_agent_loss.item()
            perf_loss += batch_perf_loss.item()
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                self.log_event(
                    "batch_complete",
                    {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": loss.item()
                    }
                )
                
        # Calculate epoch metrics
        metrics = {
            "total_loss": total_loss / len(dataloader),
            "agent_loss": agent_loss / len(dataloader),
            "perf_loss": perf_loss / len(dataloader)
        }
        
        return metrics
        
    def validate(self,
                dataloader: DataLoader) -> Dict[str, float]:
        """Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        agent_loss = 0.0
        perf_loss = 0.0
        
        with torch.no_grad():
            for text, metrics, feedback in dataloader:
                # Forward pass
                agent_scores, perf_pred = self.model(text, metrics, feedback)
                
                # Calculate losses
                agent_criterion = nn.BCEWithLogitsLoss()
                perf_criterion = nn.MSELoss()
                
                batch_agent_loss = agent_criterion(
                    agent_scores,
                    torch.ones_like(agent_scores)
                )
                
                batch_perf_loss = perf_criterion(
                    perf_pred,
                    metrics
                )
                
                # Update metrics
                total_loss += (batch_agent_loss + batch_perf_loss).item()
                agent_loss += batch_agent_loss.item()
                perf_loss += batch_perf_loss.item()
                
        # Calculate validation metrics
        metrics = {
            "val_total_loss": total_loss / len(dataloader),
            "val_agent_loss": agent_loss / len(dataloader),
            "val_perf_loss": perf_loss / len(dataloader)
        }
        
        return metrics
        
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              checkpoint_dir: str) -> Dict[str, List[float]]:
        """Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            checkpoint_dir: Checkpoint directory
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        best_val_loss = float('inf')
        
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id
        ) as run:
            # Log parameters
            mlflow.log_params({
                "num_epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            })
            
            for epoch in range(num_epochs):
                # Training
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_metrics = self.validate(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_metrics["val_total_loss"])
                
                # Update history
                history["train_loss"].append(train_metrics["total_loss"])
                history["val_loss"].append(val_metrics["val_total_loss"])
                history["learning_rate"].append(
                    self.optimizer.param_groups[0]["lr"]
                )
                
                # Log metrics
                mlflow.log_metrics({
                    **train_metrics,
                    **val_metrics,
                    "learning_rate": history["learning_rate"][-1]
                }, step=epoch)
                
                # Save checkpoint if best model
                if val_metrics["val_total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_total_loss"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "val_loss": best_val_loss
                        },
                        f"{checkpoint_dir}/best_model.pt"
                    )
                    
                # Log progress
                self.log_event(
                    "epoch_complete",
                    {
                        "epoch": epoch,
                        "train_loss": train_metrics["total_loss"],
                        "val_loss": val_metrics["val_total_loss"]
                    }
                )
                
        return history
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Log loading
        self.log_event(
            "checkpoint_loaded",
            {
                "path": checkpoint_path,
                "epoch": checkpoint["epoch"],
                "val_loss": checkpoint["val_loss"]
            }
        )