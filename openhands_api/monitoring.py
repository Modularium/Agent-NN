# mypy: ignore-errors
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aioredis
import docker
import prometheus_client as prom
import psutil

from utils.logging_util import LoggerMixin


class OpenHandsMonitoring(LoggerMixin):
    """Monitoring system for OpenHands."""

    # Prometheus metrics
    METRICS = {
        # System metrics
        "cpu_usage": prom.Gauge("openhands_cpu_usage_percent", "CPU usage percentage"),
        "memory_usage": prom.Gauge(
            "openhands_memory_usage_bytes", "Memory usage in bytes"
        ),
        "disk_usage": prom.Gauge("openhands_disk_usage_bytes", "Disk usage in bytes"),
        # Docker metrics
        "container_count": prom.Gauge(
            "openhands_container_count", "Number of running containers", ["status"]
        ),
        "image_count": prom.Gauge("openhands_image_count", "Number of Docker images"),
        # Execution metrics
        "executions_total": prom.Counter(
            "openhands_executions_total",
            "Total number of code executions",
            ["language", "status"],
        ),
        "execution_duration": prom.Histogram(
            "openhands_execution_duration_seconds",
            "Code execution duration in seconds",
            ["language"],
        ),
        # API metrics
        "api_requests": prom.Counter(
            "openhands_api_requests_total",
            "Total number of API requests",
            ["endpoint", "method", "status"],
        ),
        "api_latency": prom.Histogram(
            "openhands_api_latency_seconds",
            "API request latency in seconds",
            ["endpoint"],
        ),
    }

    def __init__(self, redis_url: Optional[str] = None, update_interval: float = 15.0):
        """Initialize monitoring system.

        Args:
            redis_url: Redis connection URL
            update_interval: Metrics update interval in seconds
        """
        super().__init__()
        self.redis_url = redis_url or "redis://localhost"
        self.update_interval = update_interval

        # Initialize clients
        self.docker = docker.from_env()
        self.redis = None

        # Start background tasks
        self.running = False
        self.tasks = []

    async def start(self):
        """Start monitoring system."""
        if self.running:
            return

        self.running = True
        self.redis = await aioredis.from_url(self.redis_url)

        # Start update tasks
        self.tasks = [
            asyncio.create_task(self._update_system_metrics()),
            asyncio.create_task(self._update_docker_metrics()),
            asyncio.create_task(self._update_execution_metrics()),
        ]

        self.log_event("monitoring_started", {"update_interval": self.update_interval})

    async def stop(self):
        """Stop monitoring system."""
        self.running = False

        # Cancel tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)

        # Close connections
        if self.redis:
            await self.redis.close()

        self.log_event("monitoring_stopped", {})

    async def _update_system_metrics(self):
        """Update system metrics."""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.METRICS["cpu_usage"].set(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.METRICS["memory_usage"].set(memory.used)

                # Disk usage
                disk = psutil.disk_usage("/")
                self.METRICS["disk_usage"].set(disk.used)

            except Exception as e:
                self.log_error(e, {"task": "system_metrics"})

            await asyncio.sleep(self.update_interval)

    async def _update_docker_metrics(self):
        """Update Docker metrics."""
        while self.running:
            try:
                # Container counts
                containers = self.docker.containers.list(all=True)
                status_counts = {}
                for container in containers:
                    status = container.status
                    status_counts[status] = status_counts.get(status, 0) + 1

                for status, count in status_counts.items():
                    self.METRICS["container_count"].labels(status=status).set(count)

                # Image count
                images = self.docker.images.list()
                self.METRICS["image_count"].set(len(images))

            except Exception as e:
                self.log_error(e, {"task": "docker_metrics"})

            await asyncio.sleep(self.update_interval)

    async def _update_execution_metrics(self):
        """Update execution metrics."""
        while self.running:
            try:
                # Get recent executions
                executions = await self.redis.keys("execution:*")

                for key in executions:
                    execution = await self.redis.hgetall(key)

                    if execution:
                        # Update execution count
                        self.METRICS["executions_total"].labels(
                            language=execution.get("language", "unknown"),
                            status=execution.get("status", "unknown"),
                        ).inc()

                        # Update duration if completed
                        if execution.get("completed_at"):
                            try:
                                start_time = datetime.fromisoformat(
                                    execution["created_at"]
                                )
                                end_time = datetime.fromisoformat(
                                    execution["completed_at"]
                                )
                                duration = (end_time - start_time).total_seconds()

                                self.METRICS["execution_duration"].labels(
                                    language=execution.get("language", "unknown")
                                ).observe(duration)

                            except (KeyError, ValueError):
                                pass

            except Exception as e:
                self.log_error(e, {"task": "execution_metrics"})

            await asyncio.sleep(self.update_interval)

    def track_api_request(
        self, endpoint: str, method: str, status: int, duration: float
    ):
        """Track API request metrics.

        Args:
            endpoint: API endpoint
            method: HTTP method
            status: Response status code
            duration: Request duration in seconds
        """
        try:
            # Update request count
            self.METRICS["api_requests"].labels(
                endpoint=endpoint, method=method, status=status
            ).inc()

            # Update latency
            self.METRICS["api_latency"].labels(endpoint=endpoint).observe(duration)

        except Exception as e:
            self.log_error(e, {"endpoint": endpoint, "method": method})

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics.

        Returns:
            Dict[str, Any]: Metrics summary
        """
        try:
            return {
                "system": {
                    "cpu_usage": self.METRICS["cpu_usage"]._value.get(),
                    "memory_usage": self.METRICS["memory_usage"]._value.get(),
                    "disk_usage": self.METRICS["disk_usage"]._value.get(),
                },
                "docker": {
                    "containers": {
                        status: self.METRICS["container_count"]
                        .labels(status=status)
                        ._value.get()
                        for status in ["running", "exited", "created"]
                    },
                    "images": self.METRICS["image_count"]._value.get(),
                },
                "executions": {
                    "total": sum(
                        self.METRICS["executions_total"]
                        .labels(language=lang, status=status)
                        ._value.get()
                        for lang in ["python", "javascript", "typescript"]
                        for status in ["completed", "failed", "error"]
                    ),
                    "success_rate": self._calculate_success_rate(),
                },
                "api": {
                    "requests": {
                        "total": sum(
                            self.METRICS["api_requests"]
                            .labels(endpoint=endpoint, method=method, status=status)
                            ._value.get()
                            for endpoint in ["/execute", "/docker", "/compose"]
                            for method in ["GET", "POST"]
                            for status in [200, 400, 500]
                        ),
                        "error_rate": self._calculate_error_rate(),
                    }
                },
            }

        except Exception as e:
            self.log_error(e, {"task": "metrics_summary"})
            return {}

    def _calculate_success_rate(self) -> float:
        """Calculate execution success rate.

        Returns:
            float: Success rate (0-1)
        """
        try:
            total = 0
            successful = 0

            for lang in ["python", "javascript", "typescript"]:
                # Get completed executions
                completed = (
                    self.METRICS["executions_total"]
                    .labels(language=lang, status="completed")
                    ._value.get()
                )

                # Get failed executions
                failed = (
                    self.METRICS["executions_total"]
                    .labels(language=lang, status="failed")
                    ._value.get()
                )

                # Get error executions
                error = (
                    self.METRICS["executions_total"]
                    .labels(language=lang, status="error")
                    ._value.get()
                )

                total += completed + failed + error
                successful += completed

            return successful / total if total > 0 else 1.0

        except Exception:
            return 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate API error rate.

        Returns:
            float: Error rate (0-1)
        """
        try:
            total = 0
            errors = 0

            for endpoint in ["/execute", "/docker", "/compose"]:
                for method in ["GET", "POST"]:
                    # Get successful requests
                    success = (
                        self.METRICS["api_requests"]
                        .labels(endpoint=endpoint, method=method, status=200)
                        ._value.get()
                    )

                    # Get client errors
                    client_errors = (
                        self.METRICS["api_requests"]
                        .labels(endpoint=endpoint, method=method, status=400)
                        ._value.get()
                    )

                    # Get server errors
                    server_errors = (
                        self.METRICS["api_requests"]
                        .labels(endpoint=endpoint, method=method, status=500)
                        ._value.get()
                    )

                    total += success + client_errors + server_errors
                    errors += client_errors + server_errors

            return errors / total if total > 0 else 0.0

        except Exception:
            return 0.0
