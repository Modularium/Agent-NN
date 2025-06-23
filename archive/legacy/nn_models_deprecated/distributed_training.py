# deprecated – moved for cleanup in v1.0.0-beta
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import os
import json
import time
from dataclasses import dataclass
from enum import Enum
from utils.logging_util import LoggerMixin

class DistributedMode(Enum):
    """Distributed training modes."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    HYBRID = "hybrid"
    ZERO = "zero"  # ZeRO optimizer states

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    mode: DistributedMode
    world_size: int
    rank: int
    master_addr: str = "localhost"
    master_port: str = "23456"
    backend: str = "nccl"
    init_method: Optional[str] = None
    timeout: int = 1800  # seconds
    find_unused_params: bool = False

class DistributedTrainer(LoggerMixin):
    """Distributed training implementation."""
    
    def __init__(self,
                 config: DistributedConfig,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None):
        """Initialize trainer.
        
        Args:
            config: Distributed configuration
            model: PyTorch model
            optimizer: Model optimizer
            loss_fn: Loss function
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        super().__init__()
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize process group
        self._setup_distributed()
        
        # Prepare model and data loaders
        self.model = self._prepare_model()
        self.train_loader = self._prepare_dataloader(train_loader)
        if val_loader:
            self.val_loader = self._prepare_dataloader(val_loader)
            
    def _setup_distributed(self):
        """Set up distributed environment."""
        try:
            # Set environment variables
            os.environ["MASTER_ADDR"] = self.config.master_addr
            os.environ["MASTER_PORT"] = self.config.master_port
            
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method,
                    world_size=self.config.world_size,
                    rank=self.config.rank,
                    timeout=timedelta(seconds=self.config.timeout)
                )
                
            # Set device
            torch.cuda.set_device(self.config.rank)
            
        except Exception as e:
            self.log_error(e, {"config": self.config.__dict__})
            raise
            
    def _prepare_model(self) -> nn.Module:
        """Prepare model for distributed training.
        
        Returns:
            nn.Module: Prepared model
        """
        try:
            # Move model to GPU
            self.model = self.model.cuda()
            
            if self.config.mode == DistributedMode.DATA_PARALLEL:
                return DistributedDataParallel(
                    self.model,
                    device_ids=[self.config.rank],
                    output_device=self.config.rank,
                    find_unused_parameters=self.config.find_unused_params
                )
                
            elif self.config.mode == DistributedMode.MODEL_PARALLEL:
                # Implement model parallelism
                pass
                
            elif self.config.mode == DistributedMode.HYBRID:
                # Implement hybrid parallelism
                pass
                
            elif self.config.mode == DistributedMode.ZERO:
                # Implement ZeRO optimizer
                from torch.distributed.optim import ZeroRedundancyOptimizer
                self.optimizer = ZeroRedundancyOptimizer(
                    self.model.parameters(),
                    optimizer_class=type(self.optimizer),
                    **self.optimizer.defaults
                )
                return self.model
                
            return self.model
            
        except Exception as e:
            self.log_error(e)
            raise
            
    def _prepare_dataloader(self,
                          loader: DataLoader) -> DataLoader:
        """Prepare data loader for distributed training.
        
        Args:
            loader: Original data loader
            
        Returns:
            DataLoader: Distributed data loader
        """
        # Create distributed sampler
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank
        )
        
        # Create new loader with sampler
        return DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=True
        )
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Set epoch for sampler
        self.train_loader.sampler.set_epoch(self.epoch)
        
        for batch in self.train_loader:
            try:
                # Move batch to GPU
                batch = {
                    k: v.cuda() if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                outputs = self.model(batch["input"])
                loss = self.loss_fn(outputs, batch["target"])
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                self.log_error(e, {"batch": num_batches})
                continue
                
        # Calculate metrics
        avg_loss = total_loss / num_batches
        
        # All-reduce metrics across processes
        if dist.is_initialized():
            dist.all_reduce(
                torch.tensor(avg_loss).cuda(),
                op=dist.ReduceOp.SUM
            )
            avg_loss /= self.config.world_size
            
        return {"loss": avg_loss}
        
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dict[str, float]: Validation metrics
        """
        if not self.val_loader:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    # Move batch to GPU
                    batch = {
                        k: v.cuda() if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                    
                    # Forward pass
                    outputs = self.model(batch["input"])
                    loss = self.loss_fn(outputs, batch["target"])
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.log_error(e, {"batch": num_batches})
                    continue
                    
        # Calculate metrics
        avg_loss = total_loss / num_batches
        
        # All-reduce metrics across processes
        if dist.is_initialized():
            dist.all_reduce(
                torch.tensor(avg_loss).cuda(),
                op=dist.ReduceOp.SUM
            )
            avg_loss /= self.config.world_size
            
        return {"val_loss": avg_loss}
        
    def save_checkpoint(self,
                       path: str,
                       epoch: int,
                       metrics: Dict[str, float]):
        """Save training checkpoint.
        
        Args:
            path: Checkpoint path
            epoch: Current epoch
            metrics: Training metrics
        """
        if self.config.rank != 0:
            return
            
        try:
            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config.__dict__
            }
            
            torch.save(state, path)
            
        except Exception as e:
            self.log_error(e, {"path": path})
            
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint.
        
        Args:
            path: Checkpoint path
            
        Returns:
            int: Last epoch
        """
        try:
            # Load on CPU first
            state = torch.load(path, map_location="cpu")
            
            # Load model state
            if isinstance(self.model, DistributedDataParallel):
                self.model.module.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state["model"])
                
            # Load optimizer state
            self.optimizer.load_state_dict(state["optimizer"])
            
            return state["epoch"]
            
        except Exception as e:
            self.log_error(e, {"path": path})
            return 0
            
    def train(self,
             num_epochs: int,
             checkpoint_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """Train model.
        
        Args:
            num_epochs: Number of epochs
            checkpoint_dir: Optional checkpoint directory
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        try:
            for epoch in range(num_epochs):
                self.epoch = epoch
                
                # Training
                train_metrics = self.train_epoch()
                history["train_loss"].append(train_metrics["loss"])
                
                # Validation
                val_metrics = self.validate()
                if val_metrics:
                    history["val_loss"].append(val_metrics["val_loss"])
                    
                # Save checkpoint
                if checkpoint_dir and self.config.rank == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"checkpoint_{epoch}.pt"
                    )
                    self.save_checkpoint(
                        checkpoint_path,
                        epoch,
                        {**train_metrics, **val_metrics}
                    )
                    
                # Log progress
                if self.config.rank == 0:
                    self.log_event(
                        "epoch_complete",
                        {
                            "epoch": epoch,
                            "metrics": {**train_metrics, **val_metrics}
                        }
                    )
                    
                # Synchronize processes
                if dist.is_initialized():
                    dist.barrier()
                    
        except Exception as e:
            self.log_error(e)
            raise
            
        finally:
            # Clean up
            self.cleanup()
            
        return history
        
    def cleanup(self):
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
            
    def __del__(self):
        """Clean up on deletion."""
        self.cleanup()

def run_distributed(
    rank: int,
    world_size: int,
    model_fn: Callable[[], nn.Module],
    train_fn: Callable[[DistributedTrainer], None],
    config: Optional[Dict[str, Any]] = None
):
    """Run distributed training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        model_fn: Function to create model
        train_fn: Training function
        config: Optional configuration
    """
    try:
        # Create distributed config
        dist_config = DistributedConfig(
            mode=DistributedMode.DATA_PARALLEL,
            world_size=world_size,
            rank=rank,
            **(config or {})
        )
        
        # Create model and trainer
        model = model_fn()
        trainer = DistributedTrainer(dist_config, model)
        
        # Run training
        train_fn(trainer)
        
    except Exception as e:
        print(f"Error in process {rank}: {e}")
        raise
        
def launch_distributed(
    world_size: int,
    model_fn: Callable[[], nn.Module],
    train_fn: Callable[[DistributedTrainer], None],
    config: Optional[Dict[str, Any]] = None
):
    """Launch distributed training.
    
    Args:
        world_size: Number of processes
        model_fn: Function to create model
        train_fn: Training function
        config: Optional configuration
    """
    mp.spawn(
        run_distributed,
        args=(world_size, model_fn, train_fn, config),
        nprocs=world_size,
        join=True
    )