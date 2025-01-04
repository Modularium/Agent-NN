from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
import mlflow
from utils.logging_util import LoggerMixin

class LayerType(Enum):
    """Types of neural network layers."""
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"

@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    input_dim: int
    output_dim: int
    params: Dict[str, Any] = None

class DynamicLayer(nn.Module):
    """Dynamically configurable neural network layer."""
    
    def __init__(self, config: LayerConfig):
        """Initialize layer.
        
        Args:
            config: Layer configuration
        """
        super().__init__()
        self.config = config
        self.layer = self._create_layer()
        
    def _create_layer(self) -> nn.Module:
        """Create layer based on configuration.
        
        Returns:
            nn.Module: Created layer
        """
        params = self.config.params or {}
        
        if self.config.layer_type == LayerType.LINEAR:
            return nn.Sequential(
                nn.Linear(
                    self.config.input_dim,
                    self.config.output_dim
                ),
                nn.LayerNorm(self.config.output_dim),
                nn.ReLU(),
                nn.Dropout(params.get("dropout", 0.1))
            )
            
        elif self.config.layer_type == LayerType.CONV1D:
            return nn.Sequential(
                nn.Conv1d(
                    self.config.input_dim,
                    self.config.output_dim,
                    kernel_size=params.get("kernel_size", 3),
                    padding=params.get("padding", 1)
                ),
                nn.BatchNorm1d(self.config.output_dim),
                nn.ReLU(),
                nn.Dropout(params.get("dropout", 0.1))
            )
            
        elif self.config.layer_type == LayerType.ATTENTION:
            num_heads = params.get("num_heads", 4)
            return nn.MultiheadAttention(
                self.config.input_dim,
                num_heads,
                dropout=params.get("dropout", 0.1)
            )
            
        elif self.config.layer_type == LayerType.TRANSFORMER:
            return nn.TransformerEncoderLayer(
                d_model=self.config.input_dim,
                nhead=params.get("num_heads", 4),
                dim_feedforward=params.get(
                    "dim_feedforward",
                    self.config.input_dim * 4
                ),
                dropout=params.get("dropout", 0.1)
            )
            
        elif self.config.layer_type == LayerType.LSTM:
            return nn.LSTM(
                self.config.input_dim,
                self.config.output_dim,
                num_layers=params.get("num_layers", 1),
                dropout=params.get("dropout", 0.1),
                batch_first=True
            )
            
        elif self.config.layer_type == LayerType.GRU:
            return nn.GRU(
                self.config.input_dim,
                self.config.output_dim,
                num_layers=params.get("num_layers", 1),
                dropout=params.get("dropout", 0.1),
                batch_first=True
            )
            
        else:
            raise ValueError(f"Unknown layer type: {self.config.layer_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        if self.config.layer_type in [LayerType.ATTENTION]:
            # Attention layers expect different input format
            x = x.transpose(0, 1)  # (B, L, D) -> (L, B, D)
            x, _ = self.layer(x, x, x)
            x = x.transpose(0, 1)  # (L, B, D) -> (B, L, D)
            return x
        return self.layer(x)

class DynamicArchitecture(nn.Module, LoggerMixin):
    """Neural network with dynamic architecture."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: Optional[List[int]] = None):
        """Initialize network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Optional hidden layer dimensions
        """
        nn.Module.__init__(self)
        LoggerMixin.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        
        # Initialize layers
        self.layers = nn.ModuleList()
        self.layer_configs: List[LayerConfig] = []
        
        # Create initial architecture
        self._create_initial_architecture()
        
    def _create_initial_architecture(self):
        """Create initial network architecture."""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        for i in range(len(dims) - 1):
            config = LayerConfig(
                layer_type=LayerType.LINEAR,
                input_dim=dims[i],
                output_dim=dims[i + 1]
            )
            self.layer_configs.append(config)
            self.layers.append(DynamicLayer(config))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
        
    def adapt_architecture(self,
                         task_requirements: Dict[str, Any],
                         performance_metrics: Dict[str, float]):
        """Adapt architecture based on requirements and performance.
        
        Args:
            task_requirements: Task requirements
            performance_metrics: Current performance metrics
        """
        # Analyze requirements
        complexity = task_requirements.get("complexity", "medium")
        sequence_data = task_requirements.get("sequence_data", False)
        attention_needed = task_requirements.get("attention_needed", False)
        
        # Get performance thresholds
        accuracy_threshold = task_requirements.get("accuracy_threshold", 0.9)
        current_accuracy = performance_metrics.get("accuracy", 0.0)
        
        # Determine needed changes
        changes_needed = []
        
        if current_accuracy < accuracy_threshold:
            if complexity == "high":
                # Add more capacity with linear layers
                changes_needed.append({
                    "type": "add_layer",
                    "layer_type": LayerType.LINEAR,
                    "params": {"dropout": 0.3}
                })
            elif sequence_data:
                # Add recurrent layer
                changes_needed.append({
                    "type": "add_layer",
                    "layer_type": LayerType.LSTM
                })
            elif attention_needed:
                # Add linear layer instead of attention
                changes_needed.append({
                    "type": "add_layer",
                    "layer_type": LayerType.LINEAR,
                    "params": {"dropout": 0.2}
                })
                
        # Apply changes
        for change in changes_needed:
            self._apply_architecture_change(change)
            
        # Log changes
        if changes_needed:
            self.log_event(
                "architecture_adapted",
                {
                    "changes": changes_needed,
                    "metrics": performance_metrics
                }
            )
            
    def _apply_architecture_change(self, change: Dict[str, Any]):
        """Apply architecture change.
        
        Args:
            change: Change specification
        """
        if change["type"] == "add_layer":
            # Get dimensions
            prev_dim = self.layers[-1].config.output_dim
            
            # Create new layer
            config = LayerConfig(
                layer_type=change["layer_type"],
                input_dim=prev_dim,
                output_dim=prev_dim
            )
            
            # Add layer
            self.layer_configs.append(config)
            self.layers.append(DynamicLayer(config))
            
            # Add output layer if needed
            if config.output_dim != self.output_dim:
                out_config = LayerConfig(
                    layer_type=LayerType.LINEAR,
                    input_dim=prev_dim,
                    output_dim=self.output_dim
                )
                self.layer_configs.append(out_config)
                self.layers.append(DynamicLayer(out_config))
                
    def save_architecture(self, path: str):
        """Save architecture configuration.
        
        Args:
            path: Save path
        """
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "layers": [
                {
                    "type": config.layer_type.value,
                    "input_dim": config.input_dim,
                    "output_dim": config.output_dim,
                    "params": config.params
                }
                for config in self.layer_configs
            ]
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load_architecture(cls, path: str) -> 'DynamicArchitecture':
        """Load architecture from configuration.
        
        Args:
            path: Load path
            
        Returns:
            DynamicArchitecture: Loaded architecture
        """
        with open(path, "r") as f:
            config = json.load(f)
            
        # Create network
        network = cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_dims=[]  # Will be replaced
        )
        
        # Clear initial layers
        network.layers = nn.ModuleList()
        network.layer_configs = []
        
        # Create layers from config
        for layer_config in config["layers"]:
            config = LayerConfig(
                layer_type=LayerType(layer_config["type"]),
                input_dim=layer_config["input_dim"],
                output_dim=layer_config["output_dim"],
                params=layer_config["params"]
            )
            network.layer_configs.append(config)
            network.layers.append(DynamicLayer(config))
            
        return network

class ArchitectureOptimizer(LoggerMixin):
    """Optimizer for dynamic architectures."""
    
    def __init__(self,
                 model: DynamicArchitecture,
                 learning_rate: float = 1e-4):
        """Initialize optimizer.
        
        Args:
            model: Dynamic architecture model
            learning_rate: Learning rate
        """
        super().__init__()
        self.model = model
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("architecture_optimization")
        
    def train_step(self,
                  batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Unpack batch
        inputs, targets = batch
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self._compute_loss(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        metrics = {
            "loss": loss.item(),
            "accuracy": self._compute_accuracy(outputs, targets)
        }
        
        return metrics
        
    def _compute_loss(self,
                     outputs: torch.Tensor,
                     targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            torch.Tensor: Loss value
        """
        # Use appropriate loss function
        if outputs.shape == targets.shape:
            return nn.MSELoss()(outputs, targets)
        else:
            return nn.CrossEntropyLoss()(outputs, targets)
            
    def _compute_accuracy(self,
                        outputs: torch.Tensor,
                        targets: torch.Tensor) -> float:
        """Compute accuracy.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            float: Accuracy value
        """
        if outputs.shape == targets.shape:
            # Regression task
            mse = nn.MSELoss()(outputs, targets).item()
            return 1.0 / (1.0 + mse)
        else:
            # Classification task
            predictions = outputs.argmax(dim=1)
            return (predictions == targets).float().mean().item()
            
    def evaluate(self,
                val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Forward pass
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
                acc = self._compute_accuracy(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1
                
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_acc / num_batches
        }
        
        # Log metrics
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name="evaluation"
        ):
            mlflow.log_metrics(metrics)
            
        return metrics
        
    def optimize_architecture(self,
                            train_loader: torch.utils.data.DataLoader,
                            val_loader: torch.utils.data.DataLoader,
                            task_requirements: Dict[str, Any],
                            num_epochs: int = 10) -> Dict[str, Any]:
        """Optimize architecture.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            task_requirements: Task requirements
            num_epochs: Number of epochs
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        best_metrics = None
        best_state = None
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_metrics = []
            
            for batch in train_loader:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
            # Calculate average metrics
            avg_metrics = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()
            }
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Update best metrics
            if best_metrics is None or val_metrics["val_accuracy"] > best_metrics["val_accuracy"]:
                best_metrics = val_metrics
                best_state = {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict()
                }
                
            # Adapt architecture if needed
            self.model.adapt_architecture(
                task_requirements,
                val_metrics
            )
            
            # Log progress
            self.log_event(
                "epoch_complete",
                {
                    "epoch": epoch,
                    "train_metrics": avg_metrics,
                    "val_metrics": val_metrics
                }
            )
            
        # Restore best state
        self.model.load_state_dict(best_state["model_state"])
        self.optimizer.load_state_dict(best_state["optimizer_state"])
        
        return {
            "best_metrics": best_metrics,
            "final_architecture": [
                {
                    "type": config.layer_type.value,
                    "input_dim": config.input_dim,
                    "output_dim": config.output_dim,
                    "params": config.params
                }
                for config in self.model.layer_configs
            ]
        }