from typing import Dict, Any, Optional, List, Union, Callable
from utils.optional_torch import torch, TORCH_AVAILABLE

if TORCH_AVAILABLE:  # pragma: no cover - optional dependency
    import torch.distributed as dist  # type: ignore
else:  # pragma: no cover - fallback
    dist = None  # type: ignore
import psutil
import threading
import queue
import time
from datetime import datetime, timedelta
import json
import os
import logging
from dataclasses import dataclass
from enum import Enum
import mlflow
from utils.logging_util import LoggerMixin


class MetricType(Enum):
    """Types of system metrics."""

    SYSTEM = "system"
    GPU = "gpu"
    PROCESS = "process"
    NETWORK = "network"
    CUSTOM = "custom"


@dataclass
class MonitorConfig:
    """Monitoring configuration."""

    interval: float = 1.0  # seconds
    history_size: int = 3600  # 1 hour at 1s interval
    log_to_file: bool = True
    log_to_mlflow: bool = True
    alert_enabled: bool = True
    custom_metrics: Optional[Dict[str, Callable]] = None


class MetricHistory:
    """Metric history storage."""

    def __init__(self, max_size: int = 3600):
        """Initialize history.

        Args:
            max_size: Maximum history size
        """
        self.max_size = max_size
        self.values = []
        self.timestamps = []

    def add(self, value: float):
        """Add value to history.

        Args:
            value: Metric value
        """
        self.values.append(value)
        self.timestamps.append(datetime.now())

        # Remove old values
        while len(self.values) > self.max_size:
            self.values.pop(0)
            self.timestamps.pop(0)

    def get_statistics(self, window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get metric statistics.

        Args:
            window: Optional time window

        Returns:
            Dict[str, float]: Metric statistics
        """
        if not self.values:
            return {}

        # Filter by window
        if window:
            cutoff = datetime.now() - window
            values = [v for v, t in zip(self.values, self.timestamps) if t > cutoff]
        else:
            values = self.values

        if not values:
            return {}

        return {
            "current": values[-1],
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }


class SystemMonitor(LoggerMixin):
    """System monitoring tool."""

    def __init__(self, config: MonitorConfig):
        """Initialize monitor.

        Args:
            config: Monitoring configuration
        """
        super().__init__()
        self.config = config

        # Initialize metrics storage
        self.metrics: Dict[str, MetricHistory] = {}

        # Initialize monitoring
        self.active = False
        self.monitor_thread = None

        # Initialize MLflow
        if config.log_to_mlflow:
            self.experiment = mlflow.set_experiment("system_monitoring")

        # Set up logging
        if config.log_to_file:
            self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = "logs/monitoring"
        os.makedirs(log_dir, exist_ok=True)

        handler = logging.FileHandler(os.path.join(log_dir, "system_metrics.log"))
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

    def start(self):
        """Start monitoring."""
        if self.active:
            return

        self.active = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

        self.log_event("monitoring_started", {"config": self.config.__dict__})

    def stop(self):
        """Stop monitoring."""
        if not self.active:
            return

        self.active = False

        if self.monitor_thread:
            self.monitor_thread.join()

        self.log_event("monitoring_stopped")

    def _monitor_loop(self):
        """Monitoring loop."""
        while self.active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()

                # Store metrics
                self._store_metrics(metrics)

                # Log metrics
                self._log_metrics(metrics)

                # Check alerts
                if self.config.alert_enabled:
                    self._check_alerts(metrics)

                # Wait for next iteration
                time.sleep(self.config.interval)

            except Exception as e:
                self.log_error(e)
                time.sleep(1.0)

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect system metrics.

        Returns:
            Dict[str, float]: System metrics
        """
        metrics = {}

        # System metrics
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        metrics.update(
            {
                "system.cpu": cpu,
                "system.memory": memory.percent,
                "system.disk": disk.percent,
            }
        )

        # GPU metrics
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)

                metrics.update(
                    {
                        f"gpu.{i}.memory_allocated": memory_allocated,
                        f"gpu.{i}.memory_reserved": memory_reserved,
                    }
                )

                # GPU utilization (requires pynvml)
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                    metrics.update(
                        {
                            f"gpu.{i}.utilization": util.gpu,
                            f"gpu.{i}.memory_utilization": util.memory,
                        }
                    )
                except:
                    pass

        # Process metrics
        process = psutil.Process()
        metrics.update(
            {
                "process.cpu": process.cpu_percent(),
                "process.memory": process.memory_percent(),
            }
        )

        # Network metrics
        if dist.is_initialized():
            metrics["network.world_size"] = dist.get_world_size()

        # Custom metrics
        if self.config.custom_metrics:
            for name, func in self.config.custom_metrics.items():
                try:
                    value = func()
                    metrics[f"custom.{name}"] = value
                except Exception as e:
                    self.log_error(e, {"metric": name})

        return metrics

    def _store_metrics(self, metrics: Dict[str, float]):
        """Store metrics in history.

        Args:
            metrics: System metrics
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = MetricHistory(self.config.history_size)
            self.metrics[name].add(value)

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics.

        Args:
            metrics: System metrics
        """
        # Log to file
        if self.config.log_to_file:
            self.logger.info("System metrics: %s", json.dumps(metrics))

        # Log to MLflow
        if self.config.log_to_mlflow:
            with mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
            ):
                mlflow.log_metrics(metrics)

    def _check_alerts(self, metrics: Dict[str, float]):
        """Check metric alerts.

        Args:
            metrics: System metrics
        """
        # CPU alert (80%)
        if metrics["system.cpu"] > 80:
            self.log_event("high_cpu_usage", {"cpu": metrics["system.cpu"]})

        # Memory alert (90%)
        if metrics["system.memory"] > 90:
            self.log_event("high_memory_usage", {"memory": metrics["system.memory"]})

        # GPU alerts
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_key = f"gpu.{i}.memory_allocated"
                if memory_key in metrics:
                    if metrics[memory_key] > 1000:  # 1GB
                        self.log_event(
                            "high_gpu_memory",
                            {"device": i, "memory_mb": metrics[memory_key]},
                        )

    def get_metrics(
        self, window: Optional[timedelta] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get all metrics.

        Args:
            window: Optional time window

        Returns:
            Dict[str, Dict[str, float]]: All metrics
        """
        return {
            name: history.get_statistics(window)
            for name, history in self.metrics.items()
        }

    def get_metric(
        self, name: str, window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get specific metric.

        Args:
            name: Metric name
            window: Optional time window

        Returns:
            Dict[str, float]: Metric statistics
        """
        if name not in self.metrics:
            return {}

        return self.metrics[name].get_statistics(window)

    def add_custom_metric(self, name: str, func: Callable[[], float]):
        """Add custom metric.

        Args:
            name: Metric name
            func: Metric function
        """
        if not self.config.custom_metrics:
            self.config.custom_metrics = {}

        self.config.custom_metrics[name] = func

    def save_metrics(self, path: str):
        """Save metrics to file.

        Args:
            path: Save path
        """
        try:
            data = {
                name: {
                    "values": history.values,
                    "timestamps": [t.isoformat() for t in history.timestamps],
                }
                for name, history in self.metrics.items()
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.log_error(e, {"path": path})

    def load_metrics(self, path: str):
        """Load metrics from file.

        Args:
            path: Load path
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)

            for name, history_data in data.items():
                history = MetricHistory(self.config.history_size)
                history.values = history_data["values"]
                history.timestamps = [
                    datetime.fromisoformat(t) for t in history_data["timestamps"]
                ]
                self.metrics[name] = history

        except Exception as e:
            self.log_error(e, {"path": path})

    def cleanup(self):
        """Clean up resources."""
        self.stop()

    def __del__(self):
        """Clean up on deletion."""        self.cleanup()
