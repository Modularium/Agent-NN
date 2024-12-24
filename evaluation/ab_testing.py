"""A/B testing for agent system."""
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

from utils.logging_util import setup_logger

logger = setup_logger(__name__)
console = Console()

class ABTest:
    """System for running A/B tests."""
    
    def __init__(self,
                 test_name: str,
                 variant_a: str,
                 variant_b: str,
                 metrics: List[str],
                 test_dir: str = "data/ab_tests"):
        """Initialize A/B test.
        
        Args:
            test_name: Name of the test
            variant_a: Name of variant A
            variant_b: Name of variant B
            metrics: List of metrics to track
            test_dir: Directory for test data
        """
        self.test_name = test_name
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.metrics = metrics
        
        # Set up directories
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.test_dir / test_name
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize data storage
        self.data: Dict[str, List[Dict[str, Any]]] = {
            variant_a: [],
            variant_b: []
        }
        
        # Load existing data
        self._load_data()
        
    def log_observation(self,
                       variant: str,
                       metrics: Dict[str, float],
                       metadata: Optional[Dict[str, Any]] = None):
        """Log an observation.
        
        Args:
            variant: Variant name
            metrics: Metric values
            metadata: Optional metadata
        """
        if variant not in self.data:
            raise ValueError(f"Unknown variant: {variant}")
            
        # Validate metrics
        for metric in self.metrics:
            if metric not in metrics:
                raise ValueError(f"Missing metric: {metric}")
                
        observation = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            **(metadata or {})
        }
        
        self.data[variant].append(observation)
        
        # Save data
        self._save_data()
        
    def calculate_results(self,
                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """Calculate test results.
        
        Args:
            confidence_level: Confidence level for statistical tests
            
        Returns:
            Dict containing test results
        """
        results = {
            "test_name": self.test_name,
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "sample_sizes": {
                self.variant_a: len(self.data[self.variant_a]),
                self.variant_b: len(self.data[self.variant_b])
            },
            "metrics": {}
        }
        
        for metric in self.metrics:
            # Get metric values
            values_a = [
                obs["metrics"][metric]
                for obs in self.data[self.variant_a]
            ]
            values_b = [
                obs["metrics"][metric]
                for obs in self.data[self.variant_b]
            ]
            
            # Calculate statistics
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            std_a = np.std(values_a)
            std_b = np.std(values_b)
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                (std_a ** 2 + std_b ** 2) / 2
            )
            effect_size = (mean_b - mean_a) / pooled_std
            
            # Calculate confidence intervals
            ci_a = stats.t.interval(
                confidence_level,
                len(values_a) - 1,
                loc=mean_a,
                scale=stats.sem(values_a)
            )
            ci_b = stats.t.interval(
                confidence_level,
                len(values_b) - 1,
                loc=mean_b,
                scale=stats.sem(values_b)
            )
            
            # Store results
            results["metrics"][metric] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": std_a,
                "std_b": std_b,
                "difference": mean_b - mean_a,
                "percent_change": (mean_b - mean_a) / mean_a * 100,
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "confidence_intervals": {
                    self.variant_a: list(ci_a),
                    self.variant_b: list(ci_b)
                },
                "significant": p_value < (1 - confidence_level)
            }
            
        return results
        
    def plot_results(self, metric: str):
        """Plot test results for a metric.
        
        Args:
            metric: Metric to plot
        """
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")
            
        # Get metric values
        values_a = [
            obs["metrics"][metric]
            for obs in self.data[self.variant_a]
        ]
        values_b = [
            obs["metrics"][metric]
            for obs in self.data[self.variant_b]
        ]
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        sns.kdeplot(
            values_a,
            label=self.variant_a,
            fill=True
        )
        sns.kdeplot(
            values_b,
            label=self.variant_b,
            fill=True
        )
        
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = self.results_dir / f"{metric}_distribution.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        # Create box plot
        plt.figure(figsize=(8, 6))
        
        data = pd.DataFrame({
            "Variant": [self.variant_a] * len(values_a) + [self.variant_b] * len(values_b),
            metric: values_a + values_b
        })
        
        sns.boxplot(x="Variant", y=metric, data=data)
        plt.title(f"Box Plot of {metric}")
        plt.grid(True)
        
        # Save plot
        plot_path = self.results_dir / f"{metric}_boxplot.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
    def plot_time_series(self, metric: str):
        """Plot metric values over time.
        
        Args:
            metric: Metric to plot
        """
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")
            
        # Create time series data
        data_a = pd.DataFrame([
            {
                "timestamp": datetime.fromisoformat(obs["timestamp"]),
                "value": obs["metrics"][metric],
                "variant": self.variant_a
            }
            for obs in self.data[self.variant_a]
        ])
        
        data_b = pd.DataFrame([
            {
                "timestamp": datetime.fromisoformat(obs["timestamp"]),
                "value": obs["metrics"][metric],
                "variant": self.variant_b
            }
            for obs in self.data[self.variant_b]
        ])
        
        data = pd.concat([data_a, data_b])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        sns.scatterplot(
            data=data,
            x="timestamp",
            y="value",
            hue="variant",
            alpha=0.6
        )
        
        # Add trend lines
        for variant in [self.variant_a, self.variant_b]:
            variant_data = data[data["variant"] == variant]
            z = np.polyfit(
                variant_data.index,
                variant_data["value"],
                1
            )
            p = np.poly1d(z)
            plt.plot(
                variant_data.index,
                p(variant_data.index),
                linestyle='--',
                alpha=0.8
            )
            
        plt.title(f"{metric} Over Time")
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Save plot
        plot_path = self.results_dir / f"{metric}_timeseries.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
    def show_summary(self, confidence_level: float = 0.95):
        """Show test summary.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        results = self.calculate_results(confidence_level)
        
        # Print test information
        console.print(f"\n[bold]A/B Test: {self.test_name}[/bold]")
        console.print(f"Variant A: {self.variant_a}")
        console.print(f"Variant B: {self.variant_b}")
        console.print(f"Sample Sizes: A={results['sample_sizes'][self.variant_a]}, "
                     f"B={results['sample_sizes'][self.variant_b]}")
        
        # Create results table
        table = Table(title="Metric Results")
        table.add_column("Metric")
        table.add_column("Difference")
        table.add_column("% Change")
        table.add_column("P-Value")
        table.add_column("Significant")
        
        for metric, stats in results["metrics"].items():
            table.add_row(
                metric,
                f"{stats['difference']:.4f}",
                f"{stats['percent_change']:+.1f}%",
                f"{stats['p_value']:.4f}",
                "✓" if stats["significant"] else "✗"
            )
            
        console.print(table)
        
    def _save_data(self):
        """Save test data to disk."""
        try:
            file_path = self.results_dir / "test_data.json"
            with open(file_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving test data: {str(e)}")
            
    def _load_data(self):
        """Load test data from disk."""
        file_path = self.results_dir / "test_data.json"
        if not file_path.exists():
            return
            
        try:
            with open(file_path) as f:
                self.data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            
    def export_results(self, output_file: str):
        """Export test results to file.
        
        Args:
            output_file: Output file path
        """
        try:
            results = self.calculate_results()
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Exported results to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            
    def get_recommendation(self,
                          confidence_level: float = 0.95) -> Dict[str, Any]:
        """Get test recommendation.
        
        Args:
            confidence_level: Confidence level for statistical tests
            
        Returns:
            Dict containing recommendation
        """
        results = self.calculate_results(confidence_level)
        
        # Count significant improvements
        improvements = 0
        regressions = 0
        
        for stats in results["metrics"].values():
            if stats["significant"]:
                if stats["difference"] > 0:
                    improvements += 1
                else:
                    regressions += 1
                    
        # Make recommendation
        if improvements > regressions:
            recommendation = {
                "decision": "adopt_b",
                "confidence": "high" if improvements > len(self.metrics) / 2 else "medium",
                "reason": f"Variant B shows significant improvements in {improvements} metrics"
            }
        elif regressions > improvements:
            recommendation = {
                "decision": "keep_a",
                "confidence": "high" if regressions > len(self.metrics) / 2 else "medium",
                "reason": f"Variant B shows significant regressions in {regressions} metrics"
            }
        else:
            recommendation = {
                "decision": "inconclusive",
                "confidence": "low",
                "reason": "No clear winner between variants"
            }
            
        return recommendation