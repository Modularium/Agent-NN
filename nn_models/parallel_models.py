# MIGRATED FROM: archive/legacy/nn_models_deprecated/parallel_models.py
# deprecated – moved for cleanup in v1.0.0-beta
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from dataclasses import dataclass
from enum import Enum
import queue
import threading
from utils.logging_util import LoggerMixin

class ParallelMode(Enum):
    """Model parallelism modes."""
    TENSOR = "tensor"  # Split tensors across devices
    PIPELINE = "pipeline"  # Split layers across devices
    HYBRID = "hybrid"  # Combine tensor and pipeline parallelism

@dataclass
class ParallelConfig:
    """Parallelism configuration."""
    mode: ParallelMode
    devices: List[int]
    chunk_size: int = 1
    overlap_comm: bool = True
    recompute_grad: bool = True
    balance_load: bool = True
    pipeline_chunks: int = 4

class TensorParallel(nn.Module):
    """Tensor parallelism implementation."""
    
    def __init__(self,
                 module: nn.Module,
                 config: ParallelConfig):
        """Initialize tensor parallel model.
        
        Args:
            module: Base module
            config: Parallelism configuration
        """
        super().__init__()
        self.module = module
        self.config = config
        self.num_devices = len(config.devices)
        
        # Split parameters across devices
        self._split_parameters()
        
    def _split_parameters(self):
        """Split model parameters across devices."""
        # Get all parameters
        params = list(self.module.parameters())
        
        # Calculate split sizes
        split_sizes = [
            len(params) // self.num_devices
            for _ in range(self.num_devices)
        ]
        split_sizes[-1] += len(params) % self.num_devices
        
        # Split and move parameters
        start = 0
        self.param_groups = []
        for i, size in enumerate(split_sizes):
            device = torch.device(f"cuda:{self.config.devices[i]}")
            group = params[start:start + size]
            
            # Move parameters to device
            for param in group:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
                    
            self.param_groups.append(group)
            start += size
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallelism.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Split input across devices
        splits = x.chunk(self.num_devices, dim=0)
        outputs = []
        
        # Process each split
        for i, (split, params) in enumerate(zip(splits, self.param_groups)):
            device = torch.device(f"cuda:{self.config.devices[i]}")
            split = split.to(device)
            
            # Forward pass on device
            with torch.cuda.device(device):
                output = self.module(split)
                outputs.append(output)
                
        # Gather and combine outputs
        return torch.cat(outputs, dim=0)

class PipelineParallel(nn.Module):
    """Pipeline parallelism implementation."""
    
    def __init__(self,
                 layers: List[nn.Module],
                 config: ParallelConfig):
        """Initialize pipeline parallel model.
        
        Args:
            layers: Model layers
            config: Parallelism configuration
        """
        super().__init__()
        self.layers = layers
        self.config = config
        self.num_devices = len(config.devices)
        
        # Split layers across devices
        self._partition_layers()
        
        # Initialize queues for pipeline
        self.queues = self._create_queues()
        
        # Initialize events for synchronization
        self.events = {
            i: torch.cuda.Event(enable_timing=True)
            for i in range(self.num_devices)
        }
        
    def _partition_layers(self):
        """Partition layers across devices."""
        # Calculate partition sizes
        num_layers = len(self.layers)
        partition_sizes = [
            num_layers // self.num_devices
            for _ in range(self.num_devices)
        ]
        partition_sizes[-1] += num_layers % self.num_devices
        
        # Create partitions
        start = 0
        self.partitions = []
        for i, size in enumerate(partition_sizes):
            device = torch.device(f"cuda:{self.config.devices[i]}")
            partition = self.layers[start:start + size]
            
            # Move partition to device
            for layer in partition:
                layer.to(device)
                
            self.partitions.append(partition)
            start += size
            
    def _create_queues(self) -> Dict[int, queue.Queue]:
        """Create queues for pipeline stages.
        
        Returns:
            Dict[int, queue.Queue]: Stage queues
        """
        return {
            i: queue.Queue(maxsize=self.config.pipeline_chunks)
            for i in range(self.num_devices - 1)
        }
        
    def _pipeline_forward(self,
                         x: torch.Tensor,
                         stage: int) -> torch.Tensor:
        """Forward pass for pipeline stage.
        
        Args:
            x: Input tensor
            stage: Pipeline stage
            
        Returns:
            torch.Tensor: Stage output
        """
        device = torch.device(f"cuda:{self.config.devices[stage]}")
        x = x.to(device)
        
        # Process through stage layers
        for layer in self.partitions[stage]:
            x = layer(x)
            
        # Record event
        self.events[stage].record()
        
        return x
        
    def _pipeline_backward(self,
                          grad: torch.Tensor,
                          stage: int):
        """Backward pass for pipeline stage.
        
        Args:
            grad: Gradient tensor
            stage: Pipeline stage
        """
        device = torch.device(f"cuda:{self.config.devices[stage]}")
        grad = grad.to(device)
        
        # Wait for forward pass
        self.events[stage].synchronize()
        
        # Backward through stage layers
        for layer in reversed(self.partitions[stage]):
            grad = layer.backward(grad)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pipeline parallelism.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Split input into chunks
        chunks = x.chunk(self.config.pipeline_chunks, dim=0)
        outputs = []
        
        # Process chunks through pipeline
        for chunk in chunks:
            # Forward pass through stages
            stage_input = chunk
            for stage in range(self.num_devices):
                stage_output = self._pipeline_forward(stage_input, stage)
                
                if stage < self.num_devices - 1:
                    # Pass to next stage
                    self.queues[stage].put(stage_output)
                    stage_input = self.queues[stage].get()
                else:
                    # Final stage
                    outputs.append(stage_output)
                    
        # Combine chunk outputs
        return torch.cat(outputs, dim=0)

class HybridParallel(nn.Module):
    """Hybrid parallelism implementation."""
    
    def __init__(self,
                 module: nn.Module,
                 config: ParallelConfig):
        """Initialize hybrid parallel model.
        
        Args:
            module: Base module
            config: Parallelism configuration
        """
        super().__init__()
        self.module = module
        self.config = config
        
        # Split model into pipeline stages
        self.stages = self._create_stages()
        
        # Apply tensor parallelism to each stage
        self.parallel_stages = [
            TensorParallel(stage, config)
            for stage in self.stages
        ]
        
        # Create pipeline
        self.pipeline = PipelineParallel(
            self.parallel_stages,
            config
        )
        
    def _create_stages(self) -> List[nn.Module]:
        """Create pipeline stages.
        
        Returns:
            List[nn.Module]: Model stages
        """
        # Analyze model structure
        layers = list(self.module.children())
        
        # Calculate optimal split points
        # This is a simplified approach - in practice, you'd want to use
        # more sophisticated methods to balance computation and memory
        num_stages = len(self.config.devices) // 2  # Use half for pipeline
        stage_size = len(layers) // num_stages
        
        # Create stages
        stages = []
        for i in range(num_stages):
            start = i * stage_size
            end = start + stage_size if i < num_stages - 1 else len(layers)
            stage = nn.Sequential(*layers[start:end])
            stages.append(stage)
            
        return stages
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hybrid parallelism.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.pipeline(x)

class ParallelManager(LoggerMixin):
    """Manager for parallel model execution."""
    
    def __init__(self, config: ParallelConfig):
        """Initialize manager.
        
        Args:
            config: Parallelism configuration
        """
        super().__init__()
        self.config = config
        
        # Initialize process group if needed
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:23456",
                world_size=len(config.devices),
                rank=0
            )
            
    def parallelize(self,
                   model: nn.Module,
                   mode: Optional[ParallelMode] = None) -> nn.Module:
        """Parallelize model.
        
        Args:
            model: PyTorch model
            mode: Optional parallelism mode (uses config mode if None)
            
        Returns:
            nn.Module: Parallelized model
        """
        mode = mode or self.config.mode
        
        try:
            if mode == ParallelMode.TENSOR:
                return TensorParallel(model, self.config)
            elif mode == ParallelMode.PIPELINE:
                layers = list(model.children())
                return PipelineParallel(layers, self.config)
            elif mode == ParallelMode.HYBRID:
                return HybridParallel(model, self.config)
            else:
                raise ValueError(f"Unknown parallel mode: {mode}")
                
        except Exception as e:
            self.log_error(e, {
                "model": type(model).__name__,
                "mode": mode
            })
            raise
            
    def optimize_parallel(self,
                         model: nn.Module) -> nn.Module:
        """Optimize parallel model.
        
        Args:
            model: Parallel model
            
        Returns:
            nn.Module: Optimized model
        """
        try:
            # Enable gradient checkpointing if configured
            if self.config.recompute_grad:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    
            # Enable CUDA graph capture if beneficial
            # This requires careful analysis of the model structure
            # and memory patterns
            
            # Optimize communication patterns
            if self.config.overlap_comm:
                # Implement communication/computation overlap
                pass
                
            return model
            
        except Exception as e:
            self.log_error(e)
            return model
            
    def cleanup(self):
        """Clean up resources."""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                
        except Exception as e:
            self.log_error(e)
            
    def __del__(self):
        """Clean up on deletion."""
        self.cleanup()
