"""Performance monitoring for LLM models."""
import time
import psutil
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

class PerformanceMonitor:
    def __init__(self, log_dir: str = "logs/performance"):
        """Initialize performance monitor.
        
        Args:
            log_dir: Directory to store performance logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "metrics": [],
            "summary": {}
        }
        
    def start_monitoring(self, model_name: str, backend: str):
        """Start monitoring a model's performance.
        
        Args:
            model_name: Name of the model
            backend: Backend type
        """
        self.current_session.update({
            "model_name": model_name,
            "backend": backend,
            "metrics": []
        })
        
    def log_inference(self,
                     prompt: str,
                     response: str,
                     start_time: float,
                     end_time: float,
                     metadata: Dict[str, Any] = None):
        """Log a single inference event.
        
        Args:
            prompt: Input prompt
            response: Model response
            start_time: Start timestamp
            end_time: End timestamp
            metadata: Additional metadata
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "duration": end_time - start_time,
            "prompt_tokens": len(prompt.split()),
            "response_tokens": len(response.split()),
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_percent": psutil.cpu_percent(),
            **(metadata or {})
        }
        
        self.current_session["metrics"].append(metrics)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary for current session.
        
        Returns:
            Dict containing performance metrics
        """
        if not self.current_session["metrics"]:
            return {}
            
        metrics = self.current_session["metrics"]
        durations = [m["duration"] for m in metrics]
        memory_usage = [m["memory_usage"] for m in metrics]
        
        summary = {
            "total_inferences": len(metrics),
            "avg_latency": np.mean(durations),
            "p90_latency": np.percentile(durations, 90),
            "p95_latency": np.percentile(durations, 95),
            "max_latency": max(durations),
            "min_latency": min(durations),
            "avg_memory_mb": np.mean(memory_usage),
            "max_memory_mb": max(memory_usage),
            "total_duration": sum(durations)
        }
        
        self.current_session["summary"] = summary
        return summary
        
    def save_session(self):
        """Save current session to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.current_session.get("model_name", "unknown")
        
        filename = f"{timestamp}_{model_name}_performance.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.current_session, f, indent=2)
            
    def load_session(self, filepath: str) -> Dict[str, Any]:
        """Load a session from disk.
        
        Args:
            filepath: Path to session file
            
        Returns:
            Dict containing session data
        """
        with open(filepath, 'r') as f:
            return json.load(f)
            
    def show_live_metrics(self):
        """Show live performance metrics."""
        table = Table(title="Live Performance Metrics")
        table.add_column("Metric")
        table.add_column("Value")
        
        with Live(table, refresh_per_second=2) as live:
            while True:
                summary = self.get_summary()
                if not summary:
                    time.sleep(1)
                    continue
                    
                table.rows.clear()
                
                table.add_row(
                    "Total Inferences",
                    str(summary["total_inferences"])
                )
                table.add_row(
                    "Average Latency",
                    f"{summary['avg_latency']:.2f}s"
                )
                table.add_row(
                    "P95 Latency",
                    f"{summary['p95_latency']:.2f}s"
                )
                table.add_row(
                    "Memory Usage",
                    f"{summary['avg_memory_mb']:.1f}MB"
                )
                table.add_row(
                    "CPU Usage",
                    f"{psutil.cpu_percent()}%"
                )
                
                live.update(table)
                time.sleep(0.5)
                
    def compare_models(self, model_logs: List[str]):
        """Compare performance across different models.
        
        Args:
            model_logs: List of paths to model performance logs
        """
        table = Table(title="Model Performance Comparison")
        table.add_column("Model")
        table.add_column("Backend")
        table.add_column("Avg Latency")
        table.add_column("P95 Latency")
        table.add_column("Memory Usage")
        table.add_column("Total Inferences")
        
        for log_path in model_logs:
            session = self.load_session(log_path)
            summary = session.get("summary", {})
            
            if summary:
                table.add_row(
                    session.get("model_name", "unknown"),
                    session.get("backend", "unknown"),
                    f"{summary.get('avg_latency', 0):.2f}s",
                    f"{summary.get('p95_latency', 0):.2f}s",
                    f"{summary.get('avg_memory_mb', 0):.1f}MB",
                    str(summary.get('total_inferences', 0))
                )
                
        console.print(table)
        
    def generate_report(self, output_file: str = None):
        """Generate a detailed performance report.
        
        Args:
            output_file: Optional path to save the report
        """
        summary = self.get_summary()
        if not summary:
            console.print("[yellow]No metrics available for report generation")
            return
            
        report = f"""Performance Report
        
Model: {self.current_session.get('model_name', 'unknown')}
Backend: {self.current_session.get('backend', 'unknown')}
Session Start: {self.current_session['start_time']}

Performance Metrics:
------------------
Total Inferences: {summary['total_inferences']}
Average Latency: {summary['avg_latency']:.2f}s
P90 Latency: {summary['p90_latency']:.2f}s
P95 Latency: {summary['p95_latency']:.2f}s
Maximum Latency: {summary['max_latency']:.2f}s
Minimum Latency: {summary['min_latency']:.2f}s

Resource Usage:
-------------
Average Memory: {summary['avg_memory_mb']:.1f}MB
Maximum Memory: {summary['max_memory_mb']:.1f}MB
Total Duration: {summary['total_duration']:.1f}s

Throughput: {summary['total_inferences'] / summary['total_duration']:.2f} requests/second
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        else:
            console.print(report)