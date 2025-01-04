from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.cuda as cuda
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import threading
import queue
import time
from datetime import datetime
import os
import json
import logging
from dataclasses import dataclass
from enum import Enum
import pynvml
from utils.logging_util import LoggerMixin

class GPUMode(Enum):
    """GPU operation modes."""
    SINGLE = "single"  # Single GPU
    DATA_PARALLEL = "data_parallel"  # DataParallel
    DISTRIBUTED = "distributed"  # DistributedDataParallel
    MODEL_PARALLEL = "model_parallel"  # Model parallelism
    PIPELINE = "pipeline"  # Pipeline parallelism

@dataclass
class GPUConfig:
    """GPU configuration."""
    mode: GPUMode
    devices: List[int]
    memory_fraction: float = 0.9
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    enable_cuda_graph: bool = False
    mixed_precision: bool = True
    gradient_sync: bool = True
    memory_efficient: bool = True

class GPUManager(LoggerMixin):
    """Manager for GPU operations and optimization."""
    
    def __init__(self,
                 config: Optional[GPUConfig] = None):
        """Initialize manager.
        
        Args:
            config: GPU configuration
        """
        super().__init__()
        self.config = config or GPUConfig(
            mode=GPUMode.SINGLE,
            devices=[0]
        )
        
        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self.gpu_count)
            ]
            self.has_gpu = True
        except:
            self.has_gpu = False
            self.gpu_count = 0
            self.gpu_handles = []
            
        # Initialize monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        # Initialize GPU if available
        if self.has_gpu:
            self._setup_gpu()
            
    def _setup_gpu(self):
        """Set up GPU environment."""
        try:
            # Set device configuration
            if self.config.mode == GPUMode.SINGLE:
                torch.cuda.set_device(self.config.devices[0])
            
            # Set memory fraction
            for device in self.config.devices:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory_fraction,
                    device
                )
            
            # Configure PyTorch
            if self.config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            if self.config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                
            # Initialize distributed setup if needed
            if self.config.mode in [
                GPUMode.DISTRIBUTED,
                GPUMode.MODEL_PARALLEL,
                GPUMode.PIPELINE
            ]:
                self._setup_distributed()
                
            # Log setup
            self.log_event(
                "gpu_setup",
                {
                    "mode": self.config.mode.value,
                    "devices": self.config.devices,
                    "memory_fraction": self.config.memory_fraction
                }
            )
            
        except Exception as e:
            self.log_error(e, {"config": self.config.__dict__})
            raise
            
    def _setup_distributed(self):
        """Set up distributed training."""
        try:
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    init_method="tcp://localhost:23456",
                    world_size=len(self.config.devices),
                    rank=0
                )
                
            # Set device
            local_rank = dist.get_rank()
            torch.cuda.set_device(self.config.devices[local_rank])
            
        except Exception as e:
            self.log_error(e, {"mode": "distributed"})
            raise
            
    def prepare_model(self,
                     model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model for GPU execution.
        
        Args:
            model: PyTorch model
            
        Returns:
            torch.nn.Module: Prepared model
        """
        try:
            if not self.has_gpu:
                return model
                
            # Move to GPU
            if self.config.mode == GPUMode.SINGLE:
                model = model.cuda(self.config.devices[0])
                
            elif self.config.mode == GPUMode.DATA_PARALLEL:
                model = torch.nn.DataParallel(
                    model,
                    device_ids=self.config.devices
                )
                
            elif self.config.mode == GPUMode.DISTRIBUTED:
                model = DistributedDataParallel(
                    model.cuda(self.config.devices[dist.get_rank()]),
                    device_ids=[self.config.devices[dist.get_rank()]],
                    output_device=self.config.devices[dist.get_rank()]
                )
                
            elif self.config.mode == GPUMode.MODEL_PARALLEL:
                # Implement custom model parallelism
                pass
                
            elif self.config.mode == GPUMode.PIPELINE:
                # Implement pipeline parallelism
                pass
                
            # Enable memory optimization
            if self.config.memory_efficient:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    
            return model
            
        except Exception as e:
            self.log_error(e, {"model": type(model).__name__})
            raise
            
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if not self.has_gpu:
            return
            
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Garbage collection
            import gc
            gc.collect()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
        except Exception as e:
            self.log_error(e)
            
    def start_monitoring(self,
                        interval: float = 1.0):
        """Start GPU monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if not self.has_gpu or self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop GPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self, interval: float):
        """GPU monitoring loop.
        
        Args:
            interval: Monitoring interval
        """
        while self.monitoring:
            try:
                metrics = self.get_gpu_metrics()
                self.metrics_queue.put(metrics)
                
                # Log if thresholds exceeded
                self._check_thresholds(metrics)
                
            except Exception as e:
                self.log_error(e)
                
            time.sleep(interval)
            
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check GPU metrics thresholds.
        
        Args:
            metrics: GPU metrics
        """
        # Memory threshold (90%)
        for device in metrics["memory"]:
            if device["used_percent"] > 90:
                self.log_event(
                    "gpu_memory_warning",
                    {
                        "device": device["index"],
                        "used_percent": device["used_percent"]
                    }
                )
                
        # Temperature threshold (80°C)
        for device in metrics["temperature"]:
            if device["gpu"] > 80:
                self.log_event(
                    "gpu_temperature_warning",
                    {
                        "device": device["index"],
                        "temperature": device["gpu"]
                    }
                )
                
        # Utilization threshold (95%)
        for device in metrics["utilization"]:
            if device["gpu"] > 95:
                self.log_event(
                    "gpu_utilization_warning",
                    {
                        "device": device["index"],
                        "utilization": device["gpu"]
                    }
                )
                
    def get_gpu_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get GPU metrics.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: GPU metrics
        """
        if not self.has_gpu:
            return {}
            
        try:
            metrics = {
                "utilization": [],
                "memory": [],
                "temperature": [],
                "power": [],
                "compute": [],
                "processes": []
            }
            
            for i, handle in enumerate(self.gpu_handles):
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["utilization"].append({
                    "index": i,
                    "gpu": util.gpu,
                    "memory": util.memory
                })
                
                # Get memory
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_percent = (mem.used / mem.total) * 100
                metrics["memory"].append({
                    "index": i,
                    "total": mem.total / (1024 * 1024),  # MB
                    "used": mem.used / (1024 * 1024),
                    "free": mem.free / (1024 * 1024),
                    "used_percent": used_percent
                })
                
                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["temperature"].append({
                    "index": i,
                    "gpu": temp
                })
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    metrics["power"].append({
                        "index": i,
                        "usage": power  # Watts
                    })
                except:
                    metrics["power"].append({
                        "index": i,
                        "usage": 0
                    })
                    
                # Get compute mode
                compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
                metrics["compute"].append({
                    "index": i,
                    "mode": str(compute_mode)
                })
                
                # Get processes
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    metrics["processes"].append({
                        "index": i,
                        "count": len(procs),
                        "processes": [
                            {
                                "pid": p.pid,
                                "memory": p.usedGpuMemory / (1024 * 1024)  # MB
                            }
                            for p in procs
                        ]
                    })
                except:
                    metrics["processes"].append({
                        "index": i,
                        "count": 0,
                        "processes": []
                    })
                    
            return metrics
            
        except Exception as e:
            self.log_error(e)
            return {}
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        if not self.has_gpu:
            return {}
            
        try:
            stats = {
                "allocated": [],
                "cached": [],
                "reserved": [],
                "active": []
            }
            
            for device in self.config.devices:
                # Get memory stats
                stats["allocated"].append({
                    "device": device,
                    "bytes": torch.cuda.memory_allocated(device),
                    "mb": torch.cuda.memory_allocated(device) / (1024 * 1024)
                })
                
                stats["cached"].append({
                    "device": device,
                    "bytes": torch.cuda.memory_reserved(device),
                    "mb": torch.cuda.memory_reserved(device) / (1024 * 1024)
                })
                
                # Get memory snapshot
                snapshot = torch.cuda.memory_snapshot()
                active_blocks = [
                    block for block in snapshot
                    if block["active"] and block["device"] == device
                ]
                
                stats["active"].append({
                    "device": device,
                    "blocks": len(active_blocks),
                    "bytes": sum(b["size"] for b in active_blocks),
                    "mb": sum(b["size"] for b in active_blocks) / (1024 * 1024)
                })
                
            return stats
            
        except Exception as e:
            self.log_error(e)
            return {}
            
    def profile_memory(self,
                      model: torch.nn.Module,
                      sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profile model memory usage.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            
        Returns:
            Dict[str, Any]: Memory profile
        """
        if not self.has_gpu:
            return {}
            
        try:
            # Record initial memory
            torch.cuda.empty_cache()
            init_mem = torch.cuda.memory_allocated()
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                _ = model(sample_input)
                
            # Record peak memory
            peak_mem = torch.cuda.max_memory_allocated()
            
            # Calculate memory usage
            model_size = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            )
            
            activation_mem = peak_mem - init_mem - model_size
            
            return {
                "model_size_mb": model_size / (1024 * 1024),
                "activation_mb": activation_mem / (1024 * 1024),
                "peak_mb": peak_mem / (1024 * 1024),
                "total_mb": (model_size + activation_mem) / (1024 * 1024)
            }
            
        except Exception as e:
            self.log_error(e)
            return {}
            
    def optimize_for_inference(self,
                             model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for inference.
        
        Args:
            model: PyTorch model
            
        Returns:
            torch.nn.Module: Optimized model
        """
        if not self.has_gpu:
            return model
            
        try:
            # Use eval mode
            model.eval()
            
            # Enable CUDA graphs if configured
            if self.config.enable_cuda_graph:
                # Implement CUDA graphs optimization
                pass
                
            # Optimize memory
            if self.config.memory_efficient:
                model = torch.jit.optimize_for_inference(
                    torch.jit.script(model)
                )
                
            return model
            
        except Exception as e:
            self.log_error(e)
            return model
            
    def cleanup(self):
        """Clean up GPU resources."""
        if not self.has_gpu:
            return
            
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Reset device
            torch.cuda.device_reset()
            
            # Shutdown distributed
            if dist.is_initialized():
                dist.destroy_process_group()
                
        except Exception as e:
            self.log_error(e)
            
    def __del__(self):
        """Clean up resources."""
        self.cleanup()