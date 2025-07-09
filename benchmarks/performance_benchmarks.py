from typing import Dict, Any, Optional, List, Union, Callable
from utils.optional_torch import torch, TORCH_AVAILABLE

if TORCH_AVAILABLE:  # pragma: no cover - optional dependency
    import torch.nn as nn  # type: ignore
    import torch.distributed as dist  # type: ignore
else:  # pragma: no cover - fallback
    nn = None  # type: ignore
    dist = None  # type: ignore
import numpy as np
import time
import json
import os
from dataclasses import dataclass
from enum import Enum
import mlflow
from utils.logging_util import LoggerMixin


class BenchmarkType(Enum):
    """Types of benchmarks."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    SCALING = "scaling"
    COMMUNICATION = "communication"


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    type: BenchmarkType
    batch_sizes: List[int]
    num_iterations: int = 100
    warmup_iterations: int = 10
    profile_memory: bool = True
    profile_cuda: bool = True
    save_traces: bool = True
    device_ids: List[int] = None


class BenchmarkResult:
    """Benchmark result container."""

    def __init__(self):
        """Initialize results."""
        self.throughput = []
        self.latency = []
        self.memory_usage = []
        self.gpu_usage = []
        self.start_time = None
        self.end_time = None

    def add_metric(
        self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add metric measurement.

        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        if name == "throughput":
            self.throughput.append(value)
        elif name == "latency":
            self.latency.append(value)
        elif name == "memory":
            self.memory_usage.append(value)
        elif name == "gpu":
            self.gpu_usage.append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get results summary.

        Returns:
            Dict[str, Any]: Results summary
        """
        duration = self.end_time - self.start_time

        return {
            "duration_seconds": duration,
            "throughput": {
                "mean": np.mean(self.throughput),
                "std": np.std(self.throughput),
                "p50": np.percentile(self.throughput, 50),
                "p95": np.percentile(self.throughput, 95),
                "p99": np.percentile(self.throughput, 99),
            },
            "latency_ms": {
                "mean": np.mean(self.latency) * 1000,
                "std": np.std(self.latency) * 1000,
                "p50": np.percentile(self.latency, 50) * 1000,
                "p95": np.percentile(self.latency, 95) * 1000,
                "p99": np.percentile(self.latency, 99) * 1000,
            },
            "memory_mb": {
                "mean": np.mean(self.memory_usage),
                "peak": max(self.memory_usage),
            },
            "gpu_utilization": {
                "mean": np.mean(self.gpu_usage),
                "peak": max(self.gpu_usage),
            },
        }


class Benchmarker(LoggerMixin):
    """System benchmarking tool."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmarker.

        Args:
            config: Benchmark configuration
        """
        super().__init__()
        self.config = config

        # Initialize MLflow
        self.experiment = mlflow.set_experiment("benchmarks")

        # Set up CUDA events
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

        # Initialize device IDs
        self.device_ids = config.device_ids or list(range(torch.cuda.device_count()))

    def benchmark_throughput(
        self, model: nn.Module, input_fn: Callable[[int], torch.Tensor]
    ) -> BenchmarkResult:
        """Benchmark model throughput.

        Args:
            model: PyTorch model
            input_fn: Function to generate input tensors

        Returns:
            BenchmarkResult: Benchmark results
        """
        results = BenchmarkResult()
        results.start_time = time.time()

        try:
            # Move model to GPU
            model = model.cuda()
            model.eval()

            # Run benchmarks for each batch size
            for batch_size in self.config.batch_sizes:
                # Warmup
                for _ in range(self.config.warmup_iterations):
                    with torch.no_grad():
                        x = input_fn(batch_size).cuda()
                        _ = model(x)

                torch.cuda.synchronize()

                # Benchmark
                for _ in range(self.config.num_iterations):
                    self.start_event.record()

                    with torch.no_grad():
                        x = input_fn(batch_size).cuda()
                        _ = model(x)

                    self.end_event.record()
                    torch.cuda.synchronize()

                    # Calculate metrics
                    elapsed = self.start_event.elapsed_time(self.end_event) / 1000
                    throughput = batch_size / elapsed
                    results.add_metric("throughput", throughput)
                    results.add_metric("latency", elapsed)

                    # Memory metrics
                    if self.config.profile_memory:
                        memory = torch.cuda.memory_allocated() / (1024 * 1024)
                        results.add_metric("memory", memory)

                    # GPU utilization
                    if self.config.profile_cuda:
                        for device in self.device_ids:
                            util = torch.cuda.utilization(device)
                            results.add_metric("gpu", util)

        except Exception as e:
            self.log_error(e)
            raise

        finally:
            results.end_time = time.time()

        return results

    def benchmark_scaling(
        self, model: nn.Module, input_fn: Callable[[int], torch.Tensor]
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark model scaling.

        Args:
            model: PyTorch model
            input_fn: Function to generate input tensors

        Returns:
            Dict[str, BenchmarkResult]: Results per configuration
        """
        results = {}

        try:
            # Test different GPU configurations
            for num_gpus in range(1, len(self.device_ids) + 1):
                devices = self.device_ids[:num_gpus]

                # Create data parallel model
                parallel_model = nn.DataParallel(model, device_ids=devices)

                # Run benchmark
                result = self.benchmark_throughput(parallel_model, input_fn)

                results[f"gpus_{num_gpus}"] = result

        except Exception as e:
            self.log_error(e)
            raise

        return results

    def benchmark_communication(
        self, size_mb: int = 100, repeat: int = 10
    ) -> BenchmarkResult:
        """Benchmark GPU communication.

        Args:
            size_mb: Data size in MB
            repeat: Number of repetitions

        Returns:
            BenchmarkResult: Benchmark results
        """
        if not dist.is_initialized() or len(self.device_ids) < 2:
            return BenchmarkResult()

        results = BenchmarkResult()
        results.start_time = time.time()

        try:
            # Create test tensor
            num_elements = size_mb * 1024 * 1024 // 4  # 4 bytes per float
            tensor = torch.randn(num_elements, device="cuda")

            # Benchmark all-reduce
            for _ in range(repeat):
                self.start_event.record()

                dist.all_reduce(tensor)

                self.end_event.record()
                torch.cuda.synchronize()

                # Calculate bandwidth
                elapsed = self.start_event.elapsed_time(self.end_event) / 1000
                bandwidth = size_mb / elapsed  # MB/s
                results.add_metric("throughput", bandwidth)
                results.add_metric("latency", elapsed)

        except Exception as e:
            self.log_error(e)
            raise

        finally:
            results.end_time = time.time()

        return results

    def profile_memory_usage(
        self, model: nn.Module, input_fn: Callable[[int], torch.Tensor]
    ) -> Dict[str, Any]:
        """Profile model memory usage.

        Args:
            model: PyTorch model
            input_fn: Function to generate input tensors

        Returns:
            Dict[str, Any]: Memory profile
        """
        if not self.config.profile_memory:
            return {}

        try:
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Move model to GPU
            model = model.cuda()
            initial_mem = torch.cuda.memory_allocated()

            # Profile forward pass
            batch = input_fn(self.config.batch_sizes[0]).cuda()
            with torch.no_grad():
                _ = model(batch)

            forward_mem = torch.cuda.memory_allocated()

            # Profile backward pass
            output = model(batch)
            if isinstance(output, torch.Tensor):
                output.sum().backward()

            backward_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()

            return {
                "model_size_mb": (initial_mem) / (1024 * 1024),
                "forward_mb": (forward_mem - initial_mem) / (1024 * 1024),
                "backward_mb": (backward_mem - forward_mem) / (1024 * 1024),
                "peak_mb": peak_mem / (1024 * 1024),
                "total_mb": backward_mem / (1024 * 1024),
            }

        except Exception as e:
            self.log_error(e)
            return {}

    def save_results(
        self, results: Union[BenchmarkResult, Dict[str, BenchmarkResult]], path: str
    ):
        """Save benchmark results.

        Args:
            results: Benchmark results
            path: Save path
        """
        try:
            # Convert results to dictionary
            if isinstance(results, BenchmarkResult):
                data = results.get_summary()
            else:
                data = {name: result.get_summary() for name, result in results.items()}

            # Add metadata
            data["config"] = {
                "type": self.config.type.value,
                "batch_sizes": self.config.batch_sizes,
                "num_iterations": self.config.num_iterations,
                "device_ids": self.device_ids,
            }

            # Save to file
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            # Log to MLflow
            with mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=os.path.basename(path),
            ):
                mlflow.log_metrics(self._flatten_metrics(data))
                mlflow.log_params(data["config"])

        except Exception as e:
            self.log_error(e, {"path": path})

    def _flatten_metrics(
        self, data: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, float]:
        """Flatten nested metrics dictionary.

        Args:
            data: Nested dictionary
            prefix: Metric name prefix

        Returns:
            Dict[str, float]: Flattened metrics
        """
        metrics = {}

        for key, value in data.items():
            if isinstance(value, dict):
                nested = self._flatten_metrics(
                    value, f"{prefix}{key}_" if prefix else f"{key}_"
                )
                metrics.update(nested)
            elif isinstance(value, (int, float)):
                metrics[f"{prefix}{key}"] = value
        return metrics
