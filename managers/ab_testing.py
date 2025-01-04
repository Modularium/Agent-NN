from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import torch
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
import mlflow
from scipy import stats
from dataclasses import dataclass
from enum import Enum
from utils.logging_util import LoggerMixin

class TestStatus(Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"

class SignificanceLevel(Enum):
    """Statistical significance levels."""
    LOW = 0.1
    MEDIUM = 0.05
    HIGH = 0.01

@dataclass
class Variant:
    """Test variant configuration."""
    name: str
    model: torch.nn.Module
    config: Dict[str, Any]
    traffic_split: float = 0.5

@dataclass
class TestResult:
    """A/B test result."""
    variant_name: str
    metrics: Dict[str, float]
    sample_size: int
    timestamp: str

class ABTest:
    """A/B test configuration and results."""
    
    def __init__(self,
                 test_id: str,
                 variants: List[Variant],
                 metrics: List[str],
                 min_samples: int = 1000,
                 max_duration: int = 7):  # days
        """Initialize test.
        
        Args:
            test_id: Test identifier
            variants: Test variants
            metrics: Metrics to track
            min_samples: Minimum samples per variant
            max_duration: Maximum test duration in days
        """
        self.test_id = test_id
        self.variants = variants
        self.metrics = metrics
        self.min_samples = min_samples
        self.max_duration = timedelta(days=max_duration)
        
        self.status = TestStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Results storage
        self.results: Dict[str, List[TestResult]] = {
            variant.name: []
            for variant in variants
        }
        
    @property
    def duration(self) -> Optional[timedelta]:
        """Get test duration.
        
        Returns:
            Optional[timedelta]: Test duration
        """
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Test information
        """
        return {
            "test_id": self.test_id,
            "status": self.status.value,
            "variants": [
                {
                    "name": v.name,
                    "config": v.config,
                    "traffic_split": v.traffic_split
                }
                for v in self.variants
            ],
            "metrics": self.metrics,
            "min_samples": self.min_samples,
            "max_duration_days": self.max_duration.days,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": {
                name: [r.__dict__ for r in results]
                for name, results in self.results.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTest':
        """Create from dictionary.
        
        Args:
            data: Test information
            
        Returns:
            ABTest: Test object
        """
        variants = [
            Variant(
                name=v["name"],
                model=None,  # Models need to be loaded separately
                config=v["config"],
                traffic_split=v["traffic_split"]
            )
            for v in data["variants"]
        ]
        
        test = cls(
            test_id=data["test_id"],
            variants=variants,
            metrics=data["metrics"],
            min_samples=data["min_samples"],
            max_duration=data["max_duration_days"]
        )
        
        test.status = TestStatus(data["status"])
        test.start_time = (
            datetime.fromisoformat(data["start_time"])
            if data["start_time"] else None
        )
        test.end_time = (
            datetime.fromisoformat(data["end_time"])
            if data["end_time"] else None
        )
        
        # Load results
        for variant_name, results in data["results"].items():
            test.results[variant_name] = [
                TestResult(**r) for r in results
            ]
            
        return test

class ABTestingManager(LoggerMixin):
    """Manager for A/B testing."""
    
    def __init__(self,
                 significance_level: SignificanceLevel = SignificanceLevel.MEDIUM):
        """Initialize manager.
        
        Args:
            significance_level: Statistical significance level
        """
        super().__init__()
        self.significance_level = significance_level
        
        # Store active tests
        self.active_tests: Dict[str, ABTest] = {}
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("ab_testing")
        
    def create_test(self,
                   test_id: str,
                   variants: List[Variant],
                   metrics: List[str],
                   min_samples: int = 1000,
                   max_duration: int = 7) -> ABTest:
        """Create new A/B test.
        
        Args:
            test_id: Test identifier
            variants: Test variants
            metrics: Metrics to track
            min_samples: Minimum samples per variant
            max_duration: Maximum test duration in days
            
        Returns:
            ABTest: Created test
        """
        if test_id in self.active_tests:
            raise ValueError(f"Test already exists: {test_id}")
            
        # Validate traffic split
        total_split = sum(v.traffic_split for v in variants)
        if not np.isclose(total_split, 1.0):
            raise ValueError("Traffic splits must sum to 1.0")
            
        # Create test
        test = ABTest(
            test_id=test_id,
            variants=variants,
            metrics=metrics,
            min_samples=min_samples,
            max_duration=max_duration
        )
        
        self.active_tests[test_id] = test
        
        # Log creation
        self.log_event(
            "test_created",
            {
                "test_id": test_id,
                "variants": [v.name for v in variants],
                "metrics": metrics
            }
        )
        
        return test
        
    def start_test(self, test_id: str):
        """Start A/B test.
        
        Args:
            test_id: Test identifier
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Unknown test: {test_id}")
            
        test = self.active_tests[test_id]
        if test.status != TestStatus.PENDING:
            raise ValueError(f"Test already started: {test_id}")
            
        test.status = TestStatus.RUNNING
        test.start_time = datetime.now()
        
        # Log start
        self.log_event(
            "test_started",
            {"test_id": test_id}
        )
        
    def stop_test(self, test_id: str):
        """Stop A/B test.
        
        Args:
            test_id: Test identifier
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Unknown test: {test_id}")
            
        test = self.active_tests[test_id]
        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test not running: {test_id}")
            
        test.status = TestStatus.STOPPED
        test.end_time = datetime.now()
        
        # Log stop
        self.log_event(
            "test_stopped",
            {"test_id": test_id}
        )
        
    def add_result(self,
                  test_id: str,
                  variant_name: str,
                  metrics: Dict[str, float]):
        """Add test result.
        
        Args:
            test_id: Test identifier
            variant_name: Variant name
            metrics: Result metrics
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Unknown test: {test_id}")
            
        test = self.active_tests[test_id]
        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test not running: {test_id}")
            
        if variant_name not in test.results:
            raise ValueError(f"Unknown variant: {variant_name}")
            
        # Add result
        result = TestResult(
            variant_name=variant_name,
            metrics=metrics,
            sample_size=1,
            timestamp=datetime.now().isoformat()
        )
        test.results[variant_name].append(result)
        
        # Log to MLflow
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=f"{test_id}_{variant_name}"
        ):
            mlflow.log_metrics(metrics)
            
        # Check if test should be completed
        if self._should_complete_test(test):
            self._complete_test(test)
            
    def _should_complete_test(self, test: ABTest) -> bool:
        """Check if test should be completed.
        
        Args:
            test: A/B test
            
        Returns:
            bool: Whether test should be completed
        """
        # Only check running tests
        if test.status != TestStatus.RUNNING:
            return False
            
        # Check duration
        if test.duration > test.max_duration:
            return True
            
        # Check sample sizes
        for results in test.results.values():
            if len(results) >= test.min_samples:
                return True
                
        return False
        
    def _complete_test(self, test: ABTest):
        """Complete A/B test.
        
        Args:
            test: A/B test
        """
        test.status = TestStatus.COMPLETED
        test.end_time = datetime.now()
        
        # Calculate final results
        results = self.get_test_results(test.test_id)
        
        # Log completion
        self.log_event(
            "test_completed",
            {
                "test_id": test.test_id,
                "results": results
            }
        )
        
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Dict[str, Any]: Test results
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Unknown test: {test_id}")
            
        test = self.active_tests[test_id]
        results = {
            "test_id": test_id,
            "status": test.status.value,
            "duration": str(test.duration) if test.duration else None,
            "variants": {}
        }
        
        # Calculate metrics for each variant
        for variant_name, variant_results in test.results.items():
            if not variant_results:
                continue
                
            metrics = {}
            for metric in test.metrics:
                values = [
                    r.metrics[metric]
                    for r in variant_results
                    if metric in r.metrics
                ]
                
                if values:
                    metrics[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "samples": len(values)
                    }
                    
            results["variants"][variant_name] = {
                "metrics": metrics,
                "sample_size": len(variant_results)
            }
            
        # Add statistical analysis
        if len(test.variants) == 2:
            results["analysis"] = self._analyze_ab_test(test)
            
        return results
        
    def _analyze_ab_test(self, test: ABTest) -> Dict[str, Any]:
        """Analyze A/B test results.
        
        Args:
            test: A/B test
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        analysis = {}
        variant_names = list(test.results.keys())
        
        for metric in test.metrics:
            # Get metric values
            values_a = [
                r.metrics[metric]
                for r in test.results[variant_names[0]]
                if metric in r.metrics
            ]
            values_b = [
                r.metrics[metric]
                for r in test.results[variant_names[1]]
                if metric in r.metrics
            ]
            
            if not values_a or not values_b:
                continue
                
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.std(values_a)**2 + np.std(values_b)**2) / 2
            )
            effect_size = (np.mean(values_a) - np.mean(values_b)) / pooled_std
            
            # Determine winner
            significant = p_value < self.significance_level.value
            if significant:
                winner = variant_names[0] if t_stat > 0 else variant_names[1]
            else:
                winner = None
                
            analysis[metric] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "significant": significant,
                "winner": winner
            }
            
        return analysis
        
    def save_state(self, path: str):
        """Save manager state.
        
        Args:
            path: Save path
        """
        state = {
            "significance_level": self.significance_level.value,
            "tests": {
                test_id: test.to_dict()
                for test_id, test in self.active_tests.items()
            }
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, path: str):
        """Load manager state.
        
        Args:
            path: Load path
        """
        with open(path, "r") as f:
            state = json.load(f)
            
        self.significance_level = SignificanceLevel(
            state["significance_level"]
        )
        
        self.active_tests = {
            test_id: ABTest.from_dict(test_data)
            for test_id, test_data in state["tests"].items()
        }