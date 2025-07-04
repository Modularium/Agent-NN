# MIGRATED FROM: archive/legacy/nn_models_deprecated/online_learning.py
# deprecated – moved for cleanup in v1.0.0-beta
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from collections import deque
import threading
import queue
import time
from datetime import datetime
import mlflow
from utils.logging_util import LoggerMixin

class StreamingBuffer:
    """Buffer for streaming data with reservoir sampling."""
    
    def __init__(self,
                 capacity: int,
                 feature_dims: Dict[str, int]):
        """Initialize buffer.
        
        Args:
            capacity: Buffer capacity
            feature_dims: Feature dimensions by name
        """
        self.capacity = capacity
        self.feature_dims = feature_dims
        
        # Initialize buffers
        self.buffers = {
            name: torch.zeros((capacity, dim))
            for name, dim in feature_dims.items()
        }
        
        self.count = 0
        self.lock = threading.Lock()
        
    def add(self, data: Dict[str, torch.Tensor]) -> bool:
        """Add data to buffer using reservoir sampling.
        
        Args:
            data: Data tensors by name
            
        Returns:
            bool: Whether data was added
        """
        with self.lock:
            if self.count < self.capacity:
                # Buffer not full, add directly
                idx = self.count
                added = True
            else:
                # Use reservoir sampling
                idx = np.random.randint(0, self.count + 1)
                added = idx < self.capacity
                
            if added:
                # Add data
                for name, tensor in data.items():
                    self.buffers[name][idx] = tensor
                    
            self.count += 1
            return added
            
    def get_batch(self,
                  batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get random batch from buffer.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: Batch data
        """
        with self.lock:
            if self.count == 0:
                return None
                
            # Sample indices
            size = min(batch_size, self.count)
            indices = torch.randperm(min(self.count, self.capacity))[:size]
            
            # Get batch
            return {
                name: buffer[indices]
                for name, buffer in self.buffers.items()
            }
            
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.count = 0
            for buffer in self.buffers.values():
                buffer.zero_()

class StreamingDataset(IterableDataset):
    """Dataset for streaming data."""
    
    def __init__(self,
                 data_queue: queue.Queue,
                 transform: Optional[Callable] = None):
        """Initialize dataset.
        
        Args:
            data_queue: Data queue
            transform: Optional transform function
        """
        self.data_queue = data_queue
        self.transform = transform
        
    def __iter__(self):
        while True:
            try:
                # Get data from queue
                data = self.data_queue.get(timeout=1.0)
                
                # Apply transform
                if self.transform:
                    data = self.transform(data)
                    
                yield data
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in data stream: {e}")
                continue

class AdaptiveLearningRate:
    """Adaptive learning rate with momentum."""
    
    def __init__(self,
                 init_lr: float = 1e-4,
                 min_lr: float = 1e-6,
                 max_lr: float = 1e-2,
                 momentum: float = 0.9):
        """Initialize adaptive learning rate.
        
        Args:
            init_lr: Initial learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            momentum: Momentum factor
        """
        self.lr = init_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.momentum = momentum
        
        self.velocity = 0.0
        self.loss_history = deque(maxlen=100)
        
    def update(self, loss: float) -> float:
        """Update learning rate based on loss.
        
        Args:
            loss: Current loss value
            
        Returns:
            float: New learning rate
        """
        # Add loss to history
        self.loss_history.append(loss)
        
        if len(self.loss_history) < 2:
            return self.lr
            
        # Calculate loss trend
        loss_diff = loss - self.loss_history[-2]
        
        # Update velocity
        self.velocity = (
            self.momentum * self.velocity +
            (1 - self.momentum) * loss_diff
        )
        
        # Adjust learning rate
        if self.velocity > 0:
            # Loss increasing, decrease lr
            self.lr *= 0.95
        else:
            # Loss decreasing, increase lr
            self.lr *= 1.05
            
        # Clip learning rate
        self.lr = max(self.min_lr, min(self.max_lr, self.lr))
        
        return self.lr

class OnlineLearner(LoggerMixin):
    """Online learning system."""
    
    def __init__(self,
                 model: nn.Module,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32,
                 update_interval: float = 0.1):
        """Initialize online learner.
        
        Args:
            model: Neural network model
            buffer_capacity: Experience buffer capacity
            batch_size: Training batch size
            update_interval: Minimum time between updates
        """
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.update_interval = update_interval
        
        # Initialize buffer
        self.buffer = StreamingBuffer(
            buffer_capacity,
            self._get_feature_dims()
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(model.parameters())
        self.adaptive_lr = AdaptiveLearningRate()
        
        # Initialize data queue
        self.data_queue = queue.Queue()
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("online_learning")
        
        # Training state
        self.running = False
        self.last_update = 0.0
        self.update_count = 0
        
    def _get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions from model.
        
        Returns:
            Dict[str, int]: Feature dimensions
        """
        # This should be implemented based on model architecture
        return {
            "input": 768,  # Example dimension
            "target": 10   # Example dimension
        }
        
    def add_data(self, data: Dict[str, torch.Tensor]):
        """Add data to processing queue.
        
        Args:
            data: Data tensors
        """
        self.data_queue.put(data)
        
    def _process_data(self):
        """Process incoming data."""
        while self.running:
            try:
                # Get data from queue
                data = self.data_queue.get(timeout=1.0)
                
                # Add to buffer
                self.buffer.add(data)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log_error(e, {"operation": "process_data"})
                
    def _update_model(self) -> Optional[Dict[str, float]]:
        """Update model with buffered data.
        
        Returns:
            Optional[Dict[str, float]]: Update metrics
        """
        # Get batch
        batch = self.buffer.get_batch(self.batch_size)
        if not batch:
            return None
            
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update learning rate
        new_lr = self.adaptive_lr.update(loss.item())
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
            
        # Update metrics
        metrics = {
            "loss": loss.item(),
            "learning_rate": new_lr
        }
        
        if isinstance(outputs, dict):
            metrics.update({
                k: v.item() if torch.is_tensor(v) else v
                for k, v in outputs.items()
                if k != "loss"
            })
            
        return metrics
        
    def _training_loop(self):
        """Main training loop."""
        while self.running:
            current_time = time.time()
            
            # Check update interval
            if current_time - self.last_update >= self.update_interval:
                # Update model
                metrics = self._update_model()
                
                if metrics:
                    # Log metrics
                    self.update_count += 1
                    
                    with mlflow.start_run(
                        experiment_id=self.experiment.experiment_id,
                        run_name=f"update_{self.update_count}"
                    ):
                        mlflow.log_metrics(metrics)
                        
                    # Log update
                    self.log_event(
                        "model_updated",
                        {
                            "update": self.update_count,
                            **metrics
                        }
                    )
                    
                self.last_update = current_time
                
    def start(self):
        """Start online learning."""
        if self.running:
            return
            
        self.running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(
            target=self._process_data
        )
        self.process_thread.start()
        
        # Start training thread
        self.train_thread = threading.Thread(
            target=self._training_loop
        )
        self.train_thread.start()
        
        # Log start
        self.log_event(
            "learning_started",
            {
                "buffer_capacity": self.buffer.capacity,
                "batch_size": self.batch_size,
                "update_interval": self.update_interval
            }
        )
        
    def stop(self):
        """Stop online learning."""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads
        self.process_thread.join()
        self.train_thread.join()
        
        # Log stop
        self.log_event(
            "learning_stopped",
            {"total_updates": self.update_count}
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics.
        
        Returns:
            Dict[str, Any]: Learning statistics
        """
        return {
            "buffer_size": self.buffer.count,
            "updates": self.update_count,
            "learning_rate": self.adaptive_lr.lr,
            "loss_trend": list(self.adaptive_lr.loss_history)
        }
        
    def save_state(self, path: str):
        """Save learner state.
        
        Args:
            path: Save path
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "adaptive_lr": {
                "lr": self.adaptive_lr.lr,
                "velocity": self.adaptive_lr.velocity,
                "loss_history": list(self.adaptive_lr.loss_history)
            },
            "update_count": self.update_count,
            "last_update": self.last_update
        }
        
        torch.save(state, path)
        
        # Log save
        self.log_event(
            "state_saved",
            {"path": path}
        )
        
    def load_state(self, path: str):
        """Load learner state.
        
        Args:
            path: Load path
        """
        state = torch.load(path)
        
        # Load states
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        
        # Load adaptive learning rate
        self.adaptive_lr.lr = state["adaptive_lr"]["lr"]
        self.adaptive_lr.velocity = state["adaptive_lr"]["velocity"]
        self.adaptive_lr.loss_history = deque(
            state["adaptive_lr"]["loss_history"],
            maxlen=100
        )
        
        # Load counters
        self.update_count = state["update_count"]
        self.last_update = state["last_update"]
        
        # Log load
        self.log_event(
            "state_loaded",
            {"path": path}
        )
