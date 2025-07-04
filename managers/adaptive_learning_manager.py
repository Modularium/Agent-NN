from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import os
import numpy as np
from utils.logging_util import LoggerMixin
from nn_models.agent_nn_v2 import TaskMetrics
import mlflow

class AdaptiveLearningManager(LoggerMixin):
    """Manager for adaptive learning and model optimization."""
    
    def __init__(self,
                 models_path: str = "models/adaptive",
                 experiment_name: str = "adaptive_learning"):
        """Initialize adaptive learning manager.
        
        Args:
            models_path: Path to model files
            experiment_name: MLflow experiment name
        """
        super().__init__()
        self.models_path = models_path
        os.makedirs(models_path, exist_ok=True)
        
        # Set up MLflow experiment
        self.experiment = mlflow.set_experiment(experiment_name)
        
        # Track experiments and variants
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.model_variants: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def create_experiment(self,
                        experiment_id: str,
                        model_id: str,
                        description: str,
                        parameters: Dict[str, Any]):
        """Create new experiment for model optimization.
        
        Args:
            experiment_id: Experiment identifier
            model_id: Base model identifier
            description: Experiment description
            parameters: Experiment parameters
        """
        self.experiments[experiment_id] = {
            "model_id": model_id,
            "description": description,
            "parameters": parameters,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "variants": []
        }
        
        # Log experiment creation
        self.log_event(
            "experiment_created",
            {
                "experiment_id": experiment_id,
                "model_id": model_id,
                "parameters": parameters
            }
        )
        
    def create_model_variant(self,
                           experiment_id: str,
                           variant_id: str,
                           architecture_changes: Dict[str, Any]):
        """Create new model variant.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier
            architecture_changes: Neural architecture changes
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        # Create variant
        self.model_variants[variant_id] = {
            "experiment_id": experiment_id,
            "architecture": architecture_changes,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Add to experiment
        self.experiments[experiment_id]["variants"].append(variant_id)
        
        # Start MLflow run
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=variant_id
        ) as run:
            mlflow.log_params({
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                **architecture_changes
            })
            
        # Log variant creation
        self.log_event(
            "variant_created",
            {
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "architecture": architecture_changes
            }
        )
        
    def update_variant_metrics(self,
                             variant_id: str,
                             metrics: Dict[str, float]):
        """Update variant performance metrics.
        
        Args:
            variant_id: Variant identifier
            metrics: Performance metrics
        """
        if variant_id not in self.model_variants:
            raise ValueError(f"Unknown variant: {variant_id}")
            
        # Update metrics
        self.model_variants[variant_id]["metrics"].update(metrics)
        self.model_variants[variant_id]["last_updated"] = datetime.now().isoformat()
        
        # Add to history
        if variant_id not in self.performance_history:
            self.performance_history[variant_id] = []
            
        self.performance_history[variant_id].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Log to MLflow
        with mlflow.start_run(run_name=variant_id):
            mlflow.log_metrics(metrics)
            
        # Log update
        self.log_event(
            "variant_metrics_updated",
            {
                "variant_id": variant_id,
                "metrics": metrics
            }
        )
        
    def get_best_variant(self,
                        experiment_id: str,
                        metric: str) -> Optional[str]:
        """Get best performing variant.
        
        Args:
            experiment_id: Experiment identifier
            metric: Metric to optimize
            
        Returns:
            Optional[str]: Best variant identifier
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        variants = self.experiments[experiment_id]["variants"]
        if not variants:
            return None
            
        # Get variant scores
        scores = []
        for variant_id in variants:
            if variant_id in self.model_variants:
                variant = self.model_variants[variant_id]
                if metric in variant["metrics"]:
                    scores.append((variant_id, variant["metrics"][metric]))
                    
        if not scores:
            return None
            
        # Return variant with best score
        return max(scores, key=lambda x: x[1])[0]
        
    def optimize_architecture(self,
                            experiment_id: str,
                            optimization_metric: str,
                            num_trials: int = 10) -> Tuple[str, Dict[str, Any]]:
        """Optimize neural architecture.
        
        Args:
            experiment_id: Experiment identifier
            optimization_metric: Metric to optimize
            num_trials: Number of optimization trials
            
        Returns:
            Tuple[str, Dict[str, Any]]: Best variant and its architecture
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        best_variant = None
        best_score = float('-inf')
        best_architecture = None
        
        for trial in range(num_trials):
            # Generate architecture changes
            architecture = self._generate_architecture_variant(
                self.experiments[experiment_id]["parameters"]
            )
            
            # Create variant
            variant_id = f"{experiment_id}_trial_{trial}"
            self.create_model_variant(
                experiment_id,
                variant_id,
                architecture
            )
            
            # Train and evaluate variant
            metrics = self._evaluate_variant(variant_id, architecture)
            self.update_variant_metrics(variant_id, metrics)
            
            # Update best variant
            if metrics[optimization_metric] > best_score:
                best_score = metrics[optimization_metric]
                best_variant = variant_id
                best_architecture = architecture
                
            # Log trial
            self.log_event(
                "optimization_trial",
                {
                    "trial": trial,
                    "variant_id": variant_id,
                    "score": metrics[optimization_metric]
                }
            )
            
        return best_variant, best_architecture
        
    def _generate_architecture_variant(self,
                                    base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate architecture variant.
        
        Args:
            base_params: Base architecture parameters
            
        Returns:
            Dict[str, Any]: Modified architecture
        """
        # Example architecture modifications:
        # - Layer sizes
        # - Activation functions
        # - Dropout rates
        # - Learning rates
        
        variant = base_params.copy()
        
        # Modify layer sizes
        if "hidden_sizes" in variant:
            variant["hidden_sizes"] = [
                size + np.random.randint(-32, 33)
                for size in variant["hidden_sizes"]
            ]
            
        # Modify dropout
        if "dropout" in variant:
            variant["dropout"] = min(1.0, max(0.0,
                variant["dropout"] + np.random.uniform(-0.1, 0.1)
            ))
            
        # Modify learning rate
        if "learning_rate" in variant:
            variant["learning_rate"] = variant["learning_rate"] * np.random.uniform(0.5, 2.0)
            
        return variant
        
    def _evaluate_variant(self,
                         variant_id: str,
                         architecture: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model variant.
        
        Args:
            variant_id: Variant identifier
            architecture: Neural architecture
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Example metrics:
        # - Loss
        # - Accuracy
        # - Response time
        # - Memory usage
        
        metrics = {
            "loss": np.random.uniform(0.1, 1.0),
            "accuracy": np.random.uniform(0.7, 1.0),
            "response_time": np.random.uniform(0.1, 0.5)
        }
        
        return metrics
        
    def get_experiment_progress(self,
                              experiment_id: str) -> Dict[str, Any]:
        """Get experiment progress and results.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dict[str, Any]: Experiment progress
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self.experiments[experiment_id]
        variants = experiment["variants"]
        
        # Calculate statistics
        stats = {
            "total_variants": len(variants),
            "completed_variants": sum(
                1 for v in variants
                if self.model_variants[v]["status"] == "completed"
            ),
            "metrics": {}
        }
        
        # Aggregate metrics
        for variant_id in variants:
            variant = self.model_variants[variant_id]
            for metric, value in variant["metrics"].items():
                if metric not in stats["metrics"]:
                    stats["metrics"][metric] = []
                stats["metrics"][metric].append(value)
                
        # Calculate metric statistics
        for metric in stats["metrics"]:
            values = stats["metrics"][metric]
            stats["metrics"][metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
            
        return stats
        
    def save_experiment(self, experiment_id: str):
        """Save experiment data.
        
        Args:
            experiment_id: Experiment identifier
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        # Save experiment data
        experiment_path = os.path.join(self.models_path, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save configuration
        config = {
            "experiment": self.experiments[experiment_id],
            "variants": {
                variant_id: self.model_variants[variant_id]
                for variant_id in self.experiments[experiment_id]["variants"]
            },
            "history": {
                variant_id: self.performance_history.get(variant_id, [])
                for variant_id in self.experiments[experiment_id]["variants"]
            }
        }
        
        with open(os.path.join(experiment_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        # Log save
        self.log_event(
            "experiment_saved",
            {
                "experiment_id": experiment_id,
                "path": experiment_path
            }
        )
        
    def load_experiment(self, experiment_id: str):
        """Load experiment data.
        
        Args:
            experiment_id: Experiment identifier
        """
        experiment_path = os.path.join(self.models_path, experiment_id)
        config_path = os.path.join(experiment_path, "config.json")
        
        if not os.path.exists(config_path):
            raise ValueError(f"No saved experiment: {experiment_id}")
            
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Restore experiment data
        self.experiments[experiment_id] = config["experiment"]
        self.model_variants.update(config["variants"])
        self.performance_history.update(config["history"])
        
        # Log load
        self.log_event(
            "experiment_loaded",
            {
                "experiment_id": experiment_id,
                "path": experiment_path
            }
        )