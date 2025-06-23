# deprecated – moved for cleanup in v1.0.0-beta
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import mlflow
from utils.logging_util import LoggerMixin

class TaskEncoder(nn.Module):
    """Task-specific feature encoder."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout: float = 0.1):
        """Initialize encoder.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Encoded features
        """
        return self.network(x)

class AttentionFusion(nn.Module):
    """Multi-head attention for feature fusion."""
    
    def __init__(self,
                 feature_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize attention fusion.
        
        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                features: List[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: List of feature tensors
            mask: Optional attention mask
            
        Returns:
            torch.Tensor: Fused features
        """
        # Stack features
        stacked = torch.stack(features, dim=0)
        
        # Self-attention
        attended, _ = self.attention(
            stacked,
            stacked,
            stacked,
            key_padding_mask=mask
        )
        
        # Residual connection and normalization
        attended = self.dropout(attended)
        fused = self.layer_norm(stacked + attended)
        
        # Average across feature sets
        return torch.mean(fused, dim=0)

class TaskHead(nn.Module):
    """Task-specific prediction head."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: float = 0.1):
        """Initialize task head.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Task predictions
        """
        return self.network(x)

class MultiTaskNetwork(nn.Module):
    """Multi-task learning network."""
    
    def __init__(self,
                 task_configs: Dict[str, Dict[str, int]],
                 shared_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize network.
        
        Args:
            task_configs: Task configurations
            shared_dim: Shared feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create task encoders
        self.encoders = nn.ModuleDict({
            task: TaskEncoder(
                config["input_dim"],
                config.get("hidden_dims", [256, 128]),
                shared_dim,
                dropout
            )
            for task, config in task_configs.items()
        })
        
        # Create attention fusion
        self.fusion = AttentionFusion(
            shared_dim,
            num_heads,
            dropout
        )
        
        # Create task heads
        self.heads = nn.ModuleDict({
            task: TaskHead(
                shared_dim,
                config.get("head_dim", 64),
                config["output_dim"],
                dropout
            )
            for task, config in task_configs.items()
        })
        
        self.task_configs = task_configs
        
    def forward(self,
                inputs: Dict[str, torch.Tensor],
                tasks: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            inputs: Input tensors by task
            tasks: Optional list of tasks to process
            
        Returns:
            Dict[str, torch.Tensor]: Task outputs
        """
        # Use all tasks if none specified
        tasks = tasks or list(self.task_configs.keys())
        
        # Encode features
        encoded = []
        for task in tasks:
            if task in inputs:
                encoded.append(
                    self.encoders[task](inputs[task])
                )
                
        # Fuse features
        fused = self.fusion(encoded)
        
        # Task-specific predictions
        outputs = {}
        for task in tasks:
            if task in inputs:
                outputs[task] = self.heads[task](fused)
                
        return outputs

class MultiTaskTrainer(LoggerMixin):
    """Multi-task learning trainer."""
    
    def __init__(self,
                 model: MultiTaskNetwork,
                 task_weights: Optional[Dict[str, float]] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """Initialize trainer.
        
        Args:
            model: Multi-task network
            task_weights: Optional task loss weights
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.model = model
        self.task_weights = task_weights or {
            task: 1.0
            for task in model.task_configs.keys()
        }
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("multi_task_learning")
        
    def _compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute task losses.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Dict[str, torch.Tensor]: Task losses
        """
        losses = {}
        
        for task, output in outputs.items():
            if task in targets:
                # Get task config
                config = self.model.task_configs[task]
                
                # Select loss function
                if config.get("loss") == "bce":
                    loss_fn = F.binary_cross_entropy_with_logits
                elif config.get("loss") == "ce":
                    loss_fn = F.cross_entropy
                else:
                    loss_fn = F.mse_loss
                    
                # Compute weighted loss
                losses[task] = self.task_weights[task] * loss_fn(
                    output,
                    targets[task]
                )
                
        return losses
        
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
        task_losses = {
            task: 0.0
            for task in self.model.task_configs.keys()
        }
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute losses
            losses = self._compute_loss(outputs, targets)
            total_batch_loss = sum(losses.values())
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            for task, loss in losses.items():
                task_losses[task] += loss.item()
                
            # Log batch
            if batch_idx % 10 == 0:
                self.log_event(
                    "batch_complete",
                    {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": total_batch_loss.item()
                    }
                )
                
        # Calculate epoch metrics
        num_batches = len(dataloader)
        metrics = {
            "total_loss": total_loss / num_batches,
            **{
                f"{task}_loss": loss / num_batches
                for task, loss in task_losses.items()
            }
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
        task_losses = {
            task: 0.0
            for task in self.model.task_configs.keys()
        }
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute losses
                losses = self._compute_loss(outputs, targets)
                total_batch_loss = sum(losses.values())
                
                # Update metrics
                total_loss += total_batch_loss.item()
                for task, loss in losses.items():
                    task_losses[task] += loss.item()
                    
        # Calculate validation metrics
        num_batches = len(dataloader)
        metrics = {
            "val_total_loss": total_loss / num_batches,
            **{
                f"val_{task}_loss": loss / num_batches
                for task, loss in task_losses.items()
            }
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
            "val_loss": []
        }
        
        best_val_loss = float('inf')
        
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id
        ) as run:
            # Log parameters
            mlflow.log_params({
                "num_epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "task_weights": self.task_weights
            })
            
            for epoch in range(num_epochs):
                # Training
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_metrics = self.validate(val_loader)
                
                # Update history
                history["train_loss"].append(train_metrics["total_loss"])
                history["val_loss"].append(val_metrics["val_total_loss"])
                
                # Log metrics
                mlflow.log_metrics({
                    **train_metrics,
                    **val_metrics
                }, step=epoch)
                
                # Save checkpoint if best model
                if val_metrics["val_total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_total_loss"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
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