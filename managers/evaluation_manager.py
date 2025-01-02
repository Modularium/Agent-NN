from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import mlflow
from utils.logging_util import LoggerMixin

class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self,
                 response_time: float,
                 token_count: int,
                 api_cost: float,
                 success_rate: float,
                 user_rating: Optional[float] = None):
        """Initialize metrics.
        
        Args:
            response_time: Response time in seconds
            token_count: Number of tokens used
            api_cost: API cost in USD
            success_rate: Success rate (0-1)
            user_rating: Optional user rating (1-5)
        """
        self.response_time = response_time
        self.token_count = token_count
        self.api_cost = api_cost
        self.success_rate = success_rate
        self.user_rating = user_rating
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Metric dictionary
        """
        return {
            "response_time": self.response_time,
            "token_count": self.token_count,
            "api_cost": self.api_cost,
            "success_rate": self.success_rate,
            "user_rating": self.user_rating,
            "timestamp": self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        """Create from dictionary.
        
        Args:
            data: Metric dictionary
            
        Returns:
            EvaluationMetrics: Metrics object
        """
        return cls(
            response_time=data["response_time"],
            token_count=data["token_count"],
            api_cost=data["api_cost"],
            success_rate=data["success_rate"],
            user_rating=data.get("user_rating")
        )

class EvaluationManager(LoggerMixin):
    """Manager for system evaluation and analysis."""
    
    def __init__(self,
                 experiment_name: str = "system_evaluation",
                 metrics_window: int = 24):  # hours
        """Initialize evaluation manager.
        
        Args:
            experiment_name: MLflow experiment name
            metrics_window: Metrics window in hours
        """
        super().__init__()
        self.metrics_window = timedelta(hours=metrics_window)
        
        # Initialize MLflow experiment
        self.experiment = mlflow.set_experiment(experiment_name)
        
        # Store metrics
        self.agent_metrics: Dict[str, List[EvaluationMetrics]] = {}
        self.system_metrics: List[Dict[str, Any]] = []
        
        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Cost tracking
        self.cost_tracking: List[Dict[str, Any]] = []
        
    def add_agent_metrics(self,
                         agent_id: str,
                         metrics: EvaluationMetrics):
        """Add agent metrics.
        
        Args:
            agent_id: Agent identifier
            metrics: Evaluation metrics
        """
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = []
            
        self.agent_metrics[agent_id].append(metrics)
        
        # Log to MLflow
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=f"agent_{agent_id}"
        ):
            mlflow.log_metrics(metrics.to_dict())
            
    def add_system_metrics(self, metrics: Dict[str, Any]):
        """Add system-wide metrics.
        
        Args:
            metrics: System metrics
        """
        metrics["timestamp"] = datetime.now().isoformat()
        self.system_metrics.append(metrics)
        
        # Log to MLflow
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name="system"
        ):
            mlflow.log_metrics(metrics)
            
    def start_ab_test(self,
                     test_id: str,
                     variants: List[str],
                     metrics: List[str],
                     duration: int) -> Dict[str, Any]:
        """Start A/B test.
        
        Args:
            test_id: Test identifier
            variants: Test variants
            metrics: Metrics to track
            duration: Test duration in hours
            
        Returns:
            Dict[str, Any]: Test configuration
        """
        test = {
            "variants": variants,
            "metrics": metrics,
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(hours=duration)).isoformat(),
            "results": {variant: [] for variant in variants},
            "status": "running"
        }
        
        self.ab_tests[test_id] = test
        
        # Log test start
        self.log_event(
            "ab_test_started",
            {
                "test_id": test_id,
                "variants": variants,
                "duration": duration
            }
        )
        
        return test
        
    def add_ab_test_result(self,
                          test_id: str,
                          variant: str,
                          metrics: Dict[str, float]):
        """Add A/B test result.
        
        Args:
            test_id: Test identifier
            variant: Test variant
            metrics: Test metrics
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"Unknown test: {test_id}")
            
        test = self.ab_tests[test_id]
        if variant not in test["variants"]:
            raise ValueError(f"Unknown variant: {variant}")
            
        # Add result
        test["results"][variant].append({
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        # Log to MLflow
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=f"ab_test_{test_id}"
        ):
            mlflow.log_metrics({
                f"{variant}_{k}": v
                for k, v in metrics.items()
            })
            
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Dict[str, Any]: Test results
        """
        if test_id not in self.ab_tests:
            raise ValueError(f"Unknown test: {test_id}")
            
        test = self.ab_tests[test_id]
        results = {
            "test_id": test_id,
            "variants": test["variants"],
            "metrics": test["metrics"],
            "duration": test["duration"],
            "status": test["status"],
            "statistics": {}
        }
        
        # Calculate statistics
        for variant in test["variants"]:
            variant_results = test["results"][variant]
            if not variant_results:
                continue
                
            stats = {}
            for metric in test["metrics"]:
                values = [
                    r["metrics"][metric]
                    for r in variant_results
                ]
                
                stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "samples": len(values)
                }
                
            results["statistics"][variant] = stats
            
        return results
        
    def get_agent_performance(self,
                            agent_id: str,
                            window: Optional[int] = None) -> Dict[str, Any]:
        """Get agent performance metrics.
        
        Args:
            agent_id: Agent identifier
            window: Optional time window in hours
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if agent_id not in self.agent_metrics:
            return {}
            
        metrics = self.agent_metrics[agent_id]
        if not metrics:
            return {}
            
        # Filter by time window
        if window:
            cutoff = datetime.now() - timedelta(hours=window)
            metrics = [
                m for m in metrics
                if m.timestamp > cutoff
            ]
            
        # Calculate statistics
        df = pd.DataFrame([m.to_dict() for m in metrics])
        stats = {
            "response_time": {
                "mean": df["response_time"].mean(),
                "p95": df["response_time"].quantile(0.95),
                "max": df["response_time"].max()
            },
            "success_rate": df["success_rate"].mean(),
            "total_cost": df["api_cost"].sum(),
            "total_tokens": df["token_count"].sum()
        }
        
        if not df["user_rating"].isna().all():
            stats["user_rating"] = {
                "mean": df["user_rating"].mean(),
                "count": df["user_rating"].count()
            }
            
        return stats
        
    def get_system_performance(self,
                             window: Optional[int] = None) -> Dict[str, Any]:
        """Get system-wide performance metrics.
        
        Args:
            window: Optional time window in hours
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = self.system_metrics
        if not metrics:
            return {}
            
        # Filter by time window
        if window:
            cutoff = datetime.now() - timedelta(hours=window)
            metrics = [
                m for m in metrics
                if datetime.fromisoformat(m["timestamp"]) > cutoff
            ]
            
        # Calculate statistics
        df = pd.DataFrame(metrics)
        stats = {}
        
        for column in df.columns:
            if column == "timestamp":
                continue
                
            stats[column] = {
                "mean": df[column].mean(),
                "std": df[column].std(),
                "min": df[column].min(),
                "max": df[column].max()
            }
            
        return stats
        
    def track_cost(self,
                  amount: float,
                  service: str,
                  details: Optional[Dict[str, Any]] = None):
        """Track API cost.
        
        Args:
            amount: Cost amount in USD
            service: Service name
            details: Optional cost details
        """
        cost = {
            "amount": amount,
            "service": service,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.cost_tracking.append(cost)
        
        # Log to MLflow
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name="costs"
        ):
            mlflow.log_metric(f"cost_{service}", amount)
            
    def get_cost_analysis(self,
                         window: Optional[int] = None) -> Dict[str, Any]:
        """Get cost analysis.
        
        Args:
            window: Optional time window in hours
            
        Returns:
            Dict[str, Any]: Cost analysis
        """
        costs = self.cost_tracking
        if not costs:
            return {}
            
        # Filter by time window
        if window:
            cutoff = datetime.now() - timedelta(hours=window)
            costs = [
                c for c in costs
                if datetime.fromisoformat(c["timestamp"]) > cutoff
            ]
            
        # Calculate statistics
        df = pd.DataFrame(costs)
        analysis = {
            "total_cost": df["amount"].sum(),
            "by_service": df.groupby("service")["amount"].sum().to_dict(),
            "daily_average": df.groupby(
                pd.to_datetime(df["timestamp"]).dt.date
            )["amount"].mean().mean()
        }
        
        return analysis
        
    def export_metrics(self, path: str):
        """Export metrics to file.
        
        Args:
            path: Export file path
        """
        data = {
            "agent_metrics": {
                agent_id: [m.to_dict() for m in metrics]
                for agent_id, metrics in self.agent_metrics.items()
            },
            "system_metrics": self.system_metrics,
            "ab_tests": self.ab_tests,
            "cost_tracking": self.cost_tracking
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
        # Log export
        self.log_event(
            "metrics_exported",
            {"path": path}
        )
        
    def import_metrics(self, path: str):
        """Import metrics from file.
        
        Args:
            path: Import file path
        """
        with open(path, "r") as f:
            data = json.load(f)
            
        # Import agent metrics
        self.agent_metrics = {
            agent_id: [
                EvaluationMetrics.from_dict(m)
                for m in metrics
            ]
            for agent_id, metrics in data["agent_metrics"].items()
        }
        
        # Import other metrics
        self.system_metrics = data["system_metrics"]
        self.ab_tests = data["ab_tests"]
        self.cost_tracking = data["cost_tracking"]
        
        # Log import
        self.log_event(
            "metrics_imported",
            {"path": path}
        )