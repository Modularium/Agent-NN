from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import torch
import numpy as np
import psutil
import threading
import queue
import time
from datetime import datetime, timedelta
import json
import asyncio
import mlflow
from dataclasses import dataclass
from enum import Enum
from collections import deque
from utils.logging_util import LoggerMixin

class MetricType(Enum):
    """Types of monitored metrics."""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    MODEL = "model"
    BUSINESS = "business"
    CUSTOM = "custom"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricConfig:
    """Configuration for a monitored metric."""
    name: str
    type: MetricType
    unit: str
    thresholds: Optional[Dict[AlertSeverity, float]] = None
    aggregation: str = "mean"  # mean, sum, min, max
    retention_days: int = 30

@dataclass
class Alert:
    """System alert."""
    metric: str
    value: float
    threshold: float
    severity: AlertSeverity
    timestamp: str
    details: Optional[Dict[str, Any]] = None

class MetricBuffer:
    """Buffer for metric values with efficient storage."""
    
    def __init__(self,
                 metric: MetricConfig,
                 max_size: int = 10000):
        """Initialize buffer.
        
        Args:
            metric: Metric configuration
            max_size: Maximum buffer size
        """
        self.metric = metric
        self.max_size = max_size
        
        # Initialize buffers
        self.values = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
        # Initialize aggregates
        self.sum = 0.0
        self.count = 0
        self.min = float('inf')
        self.max = float('-inf')
        
    def add(self, value: float):
        """Add value to buffer.
        
        Args:
            value: Metric value
        """
        self.values.append(value)
        self.timestamps.append(datetime.now())
        
        # Update aggregates
        self.sum += value
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        
    def get_statistics(self,
                      window: Optional[timedelta] = None) -> Dict[str, float]:
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
            values = [
                v for v, t in zip(self.values, self.timestamps)
                if t > cutoff
            ]
        else:
            values = list(self.values)
            
        if not values:
            return {}
            
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }
        
    def check_thresholds(self) -> Optional[Alert]:
        """Check metric thresholds.
        
        Returns:
            Optional[Alert]: Alert if threshold exceeded
        """
        if not self.metric.thresholds or not self.values:
            return None
            
        # Get latest value
        value = self.values[-1]
        
        # Check thresholds in order of severity
        for severity in [
            AlertSeverity.CRITICAL,
            AlertSeverity.ERROR,
            AlertSeverity.WARNING
        ]:
            if (severity in self.metric.thresholds and
                value >= self.metric.thresholds[severity]):
                return Alert(
                    metric=self.metric.name,
                    value=value,
                    threshold=self.metric.thresholds[severity],
                    severity=severity,
                    timestamp=datetime.now().isoformat()
                )
                
        return None

class MonitoringSystem(LoggerMixin):
    """System for comprehensive monitoring."""
    
    def __init__(self,
                 check_interval: float = 1.0,  # seconds
                 alert_handlers: Optional[List[Callable]] = None):
        """Initialize monitoring system.
        
        Args:
            check_interval: Metric check interval
            alert_handlers: Optional alert handler functions
        """
        super().__init__()
        self.check_interval = check_interval
        self.alert_handlers = alert_handlers or []
        
        # Initialize metrics
        self.metrics: Dict[str, MetricConfig] = {}
        self.buffers: Dict[str, MetricBuffer] = {}
        
        # Initialize queues
        self.metric_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        
        # Initialize state
        self.running = False
        self.threads: List[threading.Thread] = []
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("monitoring")
        
        # Add default system metrics
        self._add_system_metrics()
        
    def _add_system_metrics(self):
        """Add default system metrics."""
        system_metrics = [
            MetricConfig(
                name="cpu_usage",
                type=MetricType.SYSTEM,
                unit="percent",
                thresholds={
                    AlertSeverity.WARNING: 80.0,
                    AlertSeverity.CRITICAL: 95.0
                }
            ),
            MetricConfig(
                name="memory_usage",
                type=MetricType.SYSTEM,
                unit="percent",
                thresholds={
                    AlertSeverity.WARNING: 80.0,
                    AlertSeverity.CRITICAL: 95.0
                }
            ),
            MetricConfig(
                name="gpu_usage",
                type=MetricType.SYSTEM,
                unit="percent",
                thresholds={
                    AlertSeverity.WARNING: 80.0,
                    AlertSeverity.CRITICAL: 95.0
                }
            )
        ]
        
        for metric in system_metrics:
            self.add_metric(metric)
            
    def add_metric(self, config: MetricConfig):
        """Add metric to monitor.
        
        Args:
            config: Metric configuration
        """
        if config.name in self.metrics:
            raise ValueError(f"Metric already exists: {config.name}")
            
        self.metrics[config.name] = config
        self.buffers[config.name] = MetricBuffer(config)
        
        # Log addition
        self.log_event(
            "metric_added",
            {
                "name": config.name,
                "type": config.type.value
            }
        )
        
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler.
        
        Args:
            handler: Alert handler function
        """
        self.alert_handlers.append(handler)
        
    def record_metric(self,
                     name: str,
                     value: float,
                     details: Optional[Dict[str, Any]] = None):
        """Record metric value.
        
        Args:
            name: Metric name
            value: Metric value
            details: Optional details
        """
        if name not in self.metrics:
            raise ValueError(f"Unknown metric: {name}")
            
        self.metric_queue.put((name, value, details))
        
    def _process_metrics(self):
        """Process metric queue."""
        while self.running:
            try:
                # Get metric
                name, value, details = self.metric_queue.get(timeout=1.0)
                
                # Add to buffer
                self.buffers[name].add(value)
                
                # Check thresholds
                alert = self.buffers[name].check_thresholds()
                if alert:
                    alert.details = details
                    self.alert_queue.put(alert)
                    
                # Log to MLflow
                with mlflow.start_run(
                    experiment_id=self.experiment.experiment_id,
                    run_name=f"metric_{name}"
                ):
                    mlflow.log_metric(name, value)
                    if details:
                        mlflow.log_params(details)
                        
            except queue.Empty:
                continue
                
    def _process_alerts(self):
        """Process alert queue."""
        while self.running:
            try:
                # Get alert
                alert = self.alert_queue.get(timeout=1.0)
                
                # Handle alert
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        self.log_error(e, {
                            "handler": handler.__name__,
                            "alert": alert.__dict__
                        })
                        
                # Log alert
                self.log_event(
                    "alert_triggered",
                    {
                        "metric": alert.metric,
                        "severity": alert.severity.value,
                        "value": alert.value
                    }
                )
                
            except queue.Empty:
                continue
                
    def _collect_system_metrics(self):
        """Collect system metrics."""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.record_metric("cpu_usage", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric("memory_usage", memory.percent)
                
                # GPU usage if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.record_metric("gpu_usage", info.gpu)
                except:
                    pass
                    
                # Wait for next collection
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.log_error(e, {
                    "operation": "collect_system_metrics"
                })
                time.sleep(1.0)  # Wait before retry
                
    def start(self):
        """Start monitoring system."""
        if self.running:
            return
            
        self.running = True
        
        # Start processing threads
        self.threads = [
            threading.Thread(target=self._process_metrics),
            threading.Thread(target=self._process_alerts),
            threading.Thread(target=self._collect_system_metrics)
        ]
        
        for thread in self.threads:
            thread.start()
            
        # Log start
        self.log_event(
            "monitoring_started",
            {
                "metrics": list(self.metrics.keys()),
                "check_interval": self.check_interval
            }
        )
        
    def stop(self):
        """Stop monitoring system."""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for threads
        for thread in self.threads:
            thread.join()
            
        # Log stop
        self.log_event(
            "monitoring_stopped",
            {"metrics": list(self.metrics.keys())}
        )
        
    def get_metric_statistics(self,
                            name: str,
                            window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get metric statistics.
        
        Args:
            name: Metric name
            window: Optional time window
            
        Returns:
            Dict[str, float]: Metric statistics
        """
        if name not in self.buffers:
            raise ValueError(f"Unknown metric: {name}")
            
        return self.buffers[name].get_statistics(window)
        
    def get_all_statistics(self,
                         window: Optional[timedelta] = None) -> Dict[str, Dict[str, float]]:
        """Get all metric statistics.
        
        Args:
            window: Optional time window
            
        Returns:
            Dict[str, Dict[str, float]]: All metric statistics
        """
        return {
            name: self.get_metric_statistics(name, window)
            for name in self.metrics.keys()
        }
        
    def save_state(self, path: str):
        """Save monitoring state.
        
        Args:
            path: Save path
        """
        state = {
            "metrics": {
                name: {
                    "config": {
                        "name": config.name,
                        "type": config.type.value,
                        "unit": config.unit,
                        "thresholds": {
                            k.value: v
                            for k, v in (config.thresholds or {}).items()
                        },
                        "aggregation": config.aggregation,
                        "retention_days": config.retention_days
                    },
                    "values": list(self.buffers[name].values),
                    "timestamps": [
                        t.isoformat()
                        for t in self.buffers[name].timestamps
                    ]
                }
                for name, config in self.metrics.items()
            }
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, path: str):
        """Load monitoring state.
        
        Args:
            path: Load path
        """
        with open(path, "r") as f:
            state = json.load(f)
            
        # Clear current state
        self.metrics.clear()
        self.buffers.clear()
        
        # Load metrics
        for name, data in state["metrics"].items():
            # Create config
            config = MetricConfig(
                name=data["config"]["name"],
                type=MetricType(data["config"]["type"]),
                unit=data["config"]["unit"],
                thresholds={
                    AlertSeverity(k): v
                    for k, v in data["config"]["thresholds"].items()
                } if data["config"]["thresholds"] else None,
                aggregation=data["config"]["aggregation"],
                retention_days=data["config"]["retention_days"]
            )
            
            # Add metric
            self.add_metric(config)
            
            # Load values
            buffer = self.buffers[name]
            for value, timestamp in zip(
                data["values"],
                data["timestamps"]
            ):
                buffer.values.append(value)
                buffer.timestamps.append(
                    datetime.fromisoformat(timestamp)
                )