from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import mlflow
import os
import json
from datetime import datetime
from utils.logging_util import LoggerMixin

class GradientAccumulator:
    """Gradient accumulation for large batch training."""
    
    def __init__(self,
                 model: nn.Module,
                 accumulation_steps: int = 4):
        """Initialize accumulator.
        
        Args:
            model: Neural network model
            accumulation_steps: Number of accumulation steps
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
        # Store gradients
        self.accumulated_grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.accumulated_grads[name] = torch.zeros_like(param.data)
                
    def accumulate(self, loss: torch.Tensor):
        """Accumulate gradients.
        
        Args:
            loss: Loss tensor
        """
        # Compute gradients
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        # Accumulate gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.accumulated_grads[name].add_(param.grad.data)
                
        # Clear gradients
        self.model.zero_grad()
        
        # Update step
        self.current_step += 1
        
    def step(self):
        """Apply accumulated gradients."""
        if self.current_step > 0:
            # Apply accumulated gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.grad = self.accumulated_grads[name].clone()
                    self.accumulated_grads[name].zero_()
                    
            # Reset step counter
            self.current_step = 0
            return True
        return False

class ModelCheckpointer:
    """Model checkpointing with versioning."""
    
    def __init__(self,
                 save_dir: str,
                 model_name: str,
                 max_versions: int = 5):
        """Initialize checkpointer.
        
        Args:
            save_dir: Directory to save checkpoints
            model_name: Model name
            max_versions: Maximum versions to keep
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.max_versions = max_versions
        
        # Create directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load version history
        self.version_file = os.path.join(save_dir, "versions.json")
        self.versions = self._load_versions()
        
    def _load_versions(self) -> List[Dict[str, Any]]:
        """Load version history.
        
        Returns:
            List[Dict[str, Any]]: Version history
        """
        if os.path.exists(self.version_file):
            with open(self.version_file, "r") as f:
                return json.load(f)
        return []
        
    def _save_versions(self):
        """Save version history."""
        with open(self.version_file, "w") as f:
            json.dump(self.versions, f, indent=2)
            
    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             metrics: Dict[str, float],
             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model checkpoint.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            metrics: Performance metrics
            metadata: Optional metadata
            
        Returns:
            str: Version identifier
        """
        # Generate version ID
        version = f"v{len(self.versions) + 1}"
        timestamp = datetime.now().isoformat()
        
        # Create checkpoint
        checkpoint = {
            "version": version,
            "timestamp": timestamp,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            self.save_dir,
            f"{self.model_name}_{version}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Update version history
        self.versions.append({
            "version": version,
            "timestamp": timestamp,
            "metrics": metrics,
            "metadata": metadata or {}
        })
        
        # Remove old versions if needed
        if len(self.versions) > self.max_versions:
            old_version = self.versions.pop(0)
            old_path = os.path.join(
                self.save_dir,
                f"{self.model_name}_{old_version['version']}.pt"
            )
            if os.path.exists(old_path):
                os.remove(old_path)
                
        self._save_versions()
        return version
        
    def load(self,
             version: Optional[str] = None) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            version: Optional version to load (latest if None)
            
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        if not self.versions:
            raise ValueError("No checkpoints available")
            
        # Get version info
        if version is None:
            version_info = self.versions[-1]
        else:
            version_info = next(
                (v for v in self.versions if v["version"] == version),
                None
            )
            if not version_info:
                raise ValueError(f"Version not found: {version}")
                
        # Load checkpoint
        checkpoint_path = os.path.join(
            self.save_dir,
            f"{self.model_name}_{version_info['version']}.pt"
        )
        return torch.load(checkpoint_path)
        
    def get_best_version(self,
                        metric: str,
                        higher_better: bool = True) -> str:
        """Get best performing version.
        
        Args:
            metric: Metric to compare
            higher_better: Whether higher is better
            
        Returns:
            str: Best version identifier
        """
        if not self.versions:
            raise ValueError("No checkpoints available")
            
        # Find best version
        best_version = max(
            self.versions,
            key=lambda v: v["metrics"].get(metric, float('-inf'))
            if higher_better else
            -v["metrics"].get(metric, float('inf'))
        )
        
        return best_version["version"]

class DistributedTrainer(LoggerMixin):
    """Distributed training manager."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 accumulation_steps: int = 4,
                 max_versions: int = 5):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Validation data loader
            accumulation_steps: Gradient accumulation steps
            max_versions: Maximum checkpoint versions
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize distributed training
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{dist.get_rank()}")
        else:
            self.device = torch.device("cpu")
            
        # Wrap model
        self.model = DistributedDataParallel(
            model.to(self.device),
            device_ids=[self.device.index]
            if self.device.type == "cuda" else None
        )
        
        # Initialize gradient accumulator
        self.accumulator = GradientAccumulator(
            self.model,
            accumulation_steps
        )
        
        # Initialize checkpointer
        self.checkpointer = ModelCheckpointer(
            "checkpoints",
            model.__class__.__name__,
            max_versions
        )
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("distributed_training")
        
    def _prepare_batch(self,
                      batch: Any) -> Any:
        """Prepare batch for training.
        
        Args:
            batch: Input batch
            
        Returns:
            Any: Prepared batch
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (list, tuple)):
            return [self._prepare_batch(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._prepare_batch(v) for k, v in batch.items()}
        return batch
        
    def train_epoch(self,
                   epoch: int) -> Dict[str, float]:
        """Train one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Prepare batch
            batch = self._prepare_batch(batch)
            
            # Forward pass and loss
            outputs = self.model(batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            
            # Accumulate gradients
            self.accumulator.accumulate(loss)
            
            # Update weights if needed
            if self.accumulator.step():
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # Update metrics
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                self.log_event(
                    "batch_complete",
                    {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": loss.item()
                    }
                )
                
        return {"loss": total_loss / num_batches}
        
    def validate(self) -> Dict[str, float]:
        """Validate model.
        
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Prepare batch
                batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                
                # Update metrics
                total_loss += loss.item()
                
        return {"val_loss": total_loss / num_batches}
        
    def train(self,
              num_epochs: int,
              early_stopping: Optional[int] = None) -> Dict[str, List[float]]:
        """Train model.
        
        Args:
            num_epochs: Number of epochs
            early_stopping: Optional patience for early stopping
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id
        ) as run:
            # Log parameters
            mlflow.log_params({
                "num_epochs": num_epochs,
                "batch_size": self.train_loader.batch_size,
                "accumulation_steps": self.accumulator.accumulation_steps
            })
            
            for epoch in range(num_epochs):
                # Training
                train_metrics = self.train_epoch(epoch)
                
                # Validation
                val_metrics = self.validate()
                
                # Update history
                history["train_loss"].append(train_metrics["loss"])
                history["val_loss"].append(val_metrics["val_loss"])
                
                # Log metrics
                mlflow.log_metrics({
                    **train_metrics,
                    **val_metrics
                }, step=epoch)
                
                # Save checkpoint if best model
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                    
                    self.checkpointer.save(
                        self.model.module,
                        self.optimizer,
                        {**train_metrics, **val_metrics}
                    )
                else:
                    patience_counter += 1
                    
                # Early stopping
                if early_stopping and patience_counter >= early_stopping:
                    self.log_event(
                        "early_stopping",
                        {
                            "epoch": epoch,
                            "best_val_loss": best_val_loss
                        }
                    )
                    break
                    
                # Log progress
                self.log_event(
                    "epoch_complete",
                    {
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["val_loss"]
                    }
                )
                
        return history