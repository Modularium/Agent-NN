"""Evaluation metrics for agent system."""
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

from utils.logging_util import setup_logger

logger = setup_logger(__name__)
console = Console()

class EvaluationMetrics:
    """System for tracking and analyzing performance metrics."""
    
    def __init__(self, metrics_dir: str = "data/metrics"):
        """Initialize evaluation metrics.
        
        Args:
            metrics_dir: Directory for storing metrics
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.agent_dir = self.metrics_dir / "agents"
        self.system_dir = self.metrics_dir / "system"
        self.plots_dir = self.metrics_dir / "plots"
        
        for directory in [self.agent_dir, self.system_dir, self.plots_dir]:
            directory.mkdir(exist_ok=True)
            
        # Initialize metrics storage
        self.agent_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.system_metrics: List[Dict[str, Any]] = []
        
    def log_agent_metrics(self,
                         agent_name: str,
                         metrics: Dict[str, float],
                         metadata: Optional[Dict[str, Any]] = None):
        """Log metrics for an agent.
        
        Args:
            agent_name: Name of agent
            metrics: Performance metrics
            metadata: Optional metadata
        """
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = []
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            **(metadata or {})
        }
        
        self.agent_metrics[agent_name].append(entry)
        
        # Save metrics
        self._save_agent_metrics(agent_name)
        
    def log_system_metrics(self,
                          metrics: Dict[str, float],
                          metadata: Optional[Dict[str, Any]] = None):
        """Log system-wide metrics.
        
        Args:
            metrics: System metrics
            metadata: Optional metadata
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            **(metadata or {})
        }
        
        self.system_metrics.append(entry)
        
        # Save metrics
        self._save_system_metrics()
        
    def calculate_agent_performance(self,
                                  agent_name: str,
                                  window: str = "1d") -> Dict[str, float]:
        """Calculate agent performance metrics.
        
        Args:
            agent_name: Name of agent
            window: Time window for calculation
            
        Returns:
            Dict containing performance metrics
        """
        if agent_name not in self.agent_metrics:
            return {}
            
        # Get metrics within window
        cutoff = datetime.now() - self._parse_window(window)
        metrics = [
            entry for entry in self.agent_metrics[agent_name]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if not metrics:
            return {}
            
        # Calculate aggregate metrics
        agg_metrics = {}
        for key in metrics[0]["metrics"].keys():
            values = [m["metrics"][key] for m in metrics]
            agg_metrics.update({
                f"{key}_mean": np.mean(values),
                f"{key}_std": np.std(values),
                f"{key}_min": np.min(values),
                f"{key}_max": np.max(values)
            })
            
        return agg_metrics
        
    def calculate_system_performance(self,
                                   window: str = "1d") -> Dict[str, float]:
        """Calculate system-wide performance metrics.
        
        Args:
            window: Time window for calculation
            
        Returns:
            Dict containing performance metrics
        """
        # Get metrics within window
        cutoff = datetime.now() - self._parse_window(window)
        metrics = [
            entry for entry in self.system_metrics
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if not metrics:
            return {}
            
        # Calculate aggregate metrics
        agg_metrics = {}
        for key in metrics[0]["metrics"].keys():
            values = [m["metrics"][key] for m in metrics]
            agg_metrics.update({
                f"{key}_mean": np.mean(values),
                f"{key}_std": np.std(values),
                f"{key}_min": np.min(values),
                f"{key}_max": np.max(values)
            })
            
        return agg_metrics
        
    def plot_agent_metrics(self,
                          agent_name: str,
                          metric: str,
                          window: str = "1d"):
        """Plot agent metrics over time.
        
        Args:
            agent_name: Name of agent
            metric: Metric to plot
            window: Time window for plot
        """
        if agent_name not in self.agent_metrics:
            logger.warning(f"No metrics found for agent: {agent_name}")
            return
            
        # Get metrics within window
        cutoff = datetime.now() - self._parse_window(window)
        metrics = [
            entry for entry in self.agent_metrics[agent_name]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if not metrics:
            logger.warning(f"No metrics found in window: {window}")
            return
            
        # Create plot
        plt.figure(figsize=(10, 6))
        
        timestamps = [
            datetime.fromisoformat(m["timestamp"])
            for m in metrics
        ]
        values = [m["metrics"].get(metric, 0) for m in metrics]
        
        plt.plot(timestamps, values, marker='o')
        plt.title(f"{metric} for {agent_name}")
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.grid(True)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Save plot
        plot_path = self.plots_dir / f"{agent_name}_{metric}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
    def plot_system_metrics(self,
                          metric: str,
                          window: str = "1d"):
        """Plot system metrics over time.
        
        Args:
            metric: Metric to plot
            window: Time window for plot
        """
        # Get metrics within window
        cutoff = datetime.now() - self._parse_window(window)
        metrics = [
            entry for entry in self.system_metrics
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if not metrics:
            logger.warning(f"No metrics found in window: {window}")
            return
            
        # Create plot
        plt.figure(figsize=(10, 6))
        
        timestamps = [
            datetime.fromisoformat(m["timestamp"])
            for m in metrics
        ]
        values = [m["metrics"].get(metric, 0) for m in metrics]
        
        plt.plot(timestamps, values, marker='o')
        plt.title(f"System {metric}")
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.grid(True)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Save plot
        plot_path = self.plots_dir / f"system_{metric}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
    def plot_agent_comparison(self,
                            metric: str,
                            window: str = "1d"):
        """Plot metric comparison across agents.
        
        Args:
            metric: Metric to compare
            window: Time window for comparison
        """
        # Get metrics for each agent
        agent_data = {}
        cutoff = datetime.now() - self._parse_window(window)
        
        for agent_name in self.agent_metrics:
            metrics = [
                entry for entry in self.agent_metrics[agent_name]
                if datetime.fromisoformat(entry["timestamp"]) > cutoff
            ]
            
            if metrics:
                agent_data[agent_name] = [
                    m["metrics"].get(metric, 0)
                    for m in metrics
                ]
                
        if not agent_data:
            logger.warning(f"No metrics found in window: {window}")
            return
            
        # Create box plot
        plt.figure(figsize=(10, 6))
        
        plt.boxplot(
            list(agent_data.values()),
            labels=list(agent_data.keys())
        )
        plt.title(f"Agent Comparison: {metric}")
        plt.xlabel("Agent")
        plt.ylabel(metric)
        plt.grid(True)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Save plot
        plot_path = self.plots_dir / f"agent_comparison_{metric}.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
    def calculate_classification_metrics(self,
                                      y_true: List[int],
                                      y_pred: List[int]) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dict containing classification metrics
        """
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='binary'
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }
        
        return metrics
        
    def plot_confusion_matrix(self,
                            y_true: List[int],
                            y_pred: List[int],
                            labels: Optional[List[str]] = None):
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels or ['0', '1'],
            yticklabels=labels or ['0', '1']
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        # Save plot
        plot_path = self.plots_dir / "confusion_matrix.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
    def plot_roc_curve(self,
                      y_true: List[int],
                      y_score: List[float]):
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        plot_path = self.plots_dir / "roc_curve.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
    def _save_agent_metrics(self, agent_name: str):
        """Save agent metrics to disk.
        
        Args:
            agent_name: Name of agent
        """
        try:
            file_path = self.agent_dir / f"{agent_name}_metrics.json"
            with open(file_path, 'w') as f:
                json.dump(self.agent_metrics[agent_name], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving agent metrics: {str(e)}")
            
    def _save_system_metrics(self):
        """Save system metrics to disk."""
        try:
            file_path = self.system_dir / "system_metrics.json"
            with open(file_path, 'w') as f:
                json.dump(self.system_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving system metrics: {str(e)}")
            
    def _parse_window(self, window: str) -> timedelta:
        """Parse time window string.
        
        Args:
            window: Window string (e.g., "1d", "6h", "30m")
            
        Returns:
            timedelta object
        """
        unit = window[-1]
        value = int(window[:-1])
        
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        else:
            raise ValueError(f"Invalid window format: {window}")
            
    def show_agent_summary(self,
                          agent_name: Optional[str] = None,
                          window: str = "1d"):
        """Show agent performance summary.
        
        Args:
            agent_name: Optional agent name to filter
            window: Time window for summary
        """
        table = Table(title="Agent Performance Summary")
        table.add_column("Agent")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_column("Change")
        
        agents = (
            [agent_name] if agent_name
            else self.agent_metrics.keys()
        )
        
        for agent in agents:
            metrics = self.calculate_agent_performance(agent, window)
            if not metrics:
                continue
                
            # Calculate changes from previous window
            prev_metrics = self.calculate_agent_performance(
                agent,
                self._double_window(window)
            )
            
            for metric, value in metrics.items():
                if metric.endswith("_mean"):
                    prev_value = prev_metrics.get(metric, value)
                    change = ((value - prev_value) / prev_value * 100
                             if prev_value != 0 else 0)
                    
                    table.add_row(
                        agent,
                        metric,
                        f"{value:.4f}",
                        f"{change:+.1f}%"
                    )
                    
        console.print(table)
        
    def _double_window(self, window: str) -> str:
        """Double the time window.
        
        Args:
            window: Window string
            
        Returns:
            Doubled window string
        """
        value = int(window[:-1]) * 2
        return f"{value}{window[-1]}"