from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import os
import json
import shutil
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import tempfile
import hashlib
from utils.logging_util import LoggerMixin

class ModelVersion:
    """Model version information."""
    
    def __init__(self,
                 version_id: str,
                 model_id: str,
                 path: str,
                 metrics: Dict[str, float],
                 metadata: Dict[str, Any],
                 timestamp: str):
        """Initialize model version.
        
        Args:
            version_id: Version identifier
            model_id: Model identifier
            path: Model path
            metrics: Performance metrics
            metadata: Version metadata
            timestamp: Creation timestamp
        """
        self.version_id = version_id
        self.model_id = model_id
        self.path = path
        self.metrics = metrics
        self.metadata = metadata
        self.timestamp = timestamp
        
    @property
    def age_hours(self) -> float:
        """Get version age in hours.
        
        Returns:
            float: Age in hours
        """
        created = datetime.fromisoformat(self.timestamp)
        age = datetime.now() - created
        return age.total_seconds() / 3600
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Version information
        """
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "path": self.path,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary.
        
        Args:
            data: Version information
            
        Returns:
            ModelVersion: Version object
        """
        return cls(
            version_id=data["version_id"],
            model_id=data["model_id"],
            path=data["path"],
            metrics=data["metrics"],
            metadata=data["metadata"],
            timestamp=data["timestamp"]
        )

class ModelRegistry(LoggerMixin):
    """Registry for managing model versions."""
    
    def __init__(self,
                 registry_dir: str = "model_registry",
                 max_versions: int = 10):
        """Initialize model registry.
        
        Args:
            registry_dir: Registry directory
            max_versions: Maximum versions per model
        """
        super().__init__()
        self.registry_dir = registry_dir
        self.max_versions = max_versions
        
        # Create directories
        os.makedirs(registry_dir, exist_ok=True)
        os.makedirs(os.path.join(registry_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(registry_dir, "metadata"), exist_ok=True)
        
        # Initialize MLflow client
        self.mlflow_client = MlflowClient()
        
        # Load registry state
        self.versions: Dict[str, List[ModelVersion]] = {}
        self.load_state()
        
    def _get_model_dir(self, model_id: str) -> str:
        """Get model directory.
        
        Args:
            model_id: Model identifier
            
        Returns:
            str: Model directory path
        """
        return os.path.join(self.registry_dir, "models", model_id)
        
    def _get_metadata_path(self, model_id: str) -> str:
        """Get metadata file path.
        
        Args:
            model_id: Model identifier
            
        Returns:
            str: Metadata file path
        """
        return os.path.join(
            self.registry_dir,
            "metadata",
            f"{model_id}.json"
        )
        
    def load_state(self):
        """Load registry state."""
        # Load model versions
        metadata_dir = os.path.join(self.registry_dir, "metadata")
        for filename in os.listdir(metadata_dir):
            if filename.endswith(".json"):
                model_id = filename[:-5]
                metadata_path = os.path.join(metadata_dir, filename)
                
                with open(metadata_path, "r") as f:
                    versions_data = json.load(f)
                    
                self.versions[model_id] = [
                    ModelVersion.from_dict(v)
                    for v in versions_data
                ]
                
    def save_state(self):
        """Save registry state."""
        for model_id, versions in self.versions.items():
            metadata_path = self._get_metadata_path(model_id)
            
            with open(metadata_path, "w") as f:
                json.dump(
                    [v.to_dict() for v in versions],
                    f,
                    indent=2
                )
                
    def register_model(self,
                      model: nn.Module,
                      model_id: str,
                      metrics: Dict[str, float],
                      metadata: Optional[Dict[str, Any]] = None) -> ModelVersion:
        """Register model version.
        
        Args:
            model: Model to register
            model_id: Model identifier
            metrics: Performance metrics
            metadata: Optional metadata
            
        Returns:
            ModelVersion: Registered version
        """
        # Create model directory
        model_dir = self._get_model_dir(model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate version ID
        timestamp = datetime.now().isoformat()
        version_id = hashlib.sha256(
            f"{model_id}_{timestamp}".encode()
        ).hexdigest()[:8]
        
        # Save model
        model_path = os.path.join(model_dir, f"{version_id}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            path=model_path,
            metrics=metrics,
            metadata=metadata or {},
            timestamp=timestamp
        )
        
        # Add to versions
        if model_id not in self.versions:
            self.versions[model_id] = []
        self.versions[model_id].append(version)
        
        # Remove old versions if needed
        if len(self.versions[model_id]) > self.max_versions:
            old_version = self.versions[model_id].pop(0)
            if os.path.exists(old_version.path):
                os.remove(old_version.path)
                
        # Save state
        self.save_state()
        
        # Log to MLflow
        with mlflow.start_run() as run:
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, "model.pt")
                torch.save(model, tmp_path)
                mlflow.log_artifact(tmp_path, "model")
                
            # Log metadata
            if metadata:
                mlflow.log_params(metadata)
                
        # Log registration
        self.log_event(
            "model_registered",
            {
                "model_id": model_id,
                "version_id": version_id,
                "metrics": metrics
            }
        )
        
        return version
        
    def get_version(self,
                   model_id: str,
                   version_id: Optional[str] = None) -> Optional[ModelVersion]:
        """Get model version.
        
        Args:
            model_id: Model identifier
            version_id: Optional version identifier (latest if None)
            
        Returns:
            Optional[ModelVersion]: Model version
        """
        if model_id not in self.versions:
            return None
            
        versions = self.versions[model_id]
        if not versions:
            return None
            
        if version_id:
            # Find specific version
            for version in versions:
                if version.version_id == version_id:
                    return version
            return None
            
        # Return latest version
        return versions[-1]
        
    def load_model(self,
                  model_id: str,
                  version_id: Optional[str] = None,
                  model_class: Optional[type] = None) -> Optional[nn.Module]:
        """Load model from registry.
        
        Args:
            model_id: Model identifier
            version_id: Optional version identifier (latest if None)
            model_class: Optional model class (if not provided, loads state dict)
            
        Returns:
            Optional[nn.Module]: Loaded model
        """
        version = self.get_version(model_id, version_id)
        if not version:
            return None
            
        try:
            if model_class:
                # Create new model instance
                model = model_class()
                model.load_state_dict(torch.load(version.path))
                return model
            else:
                # Load state dict only
                return torch.load(version.path)
                
        except Exception as e:
            self.log_error(e, {
                "model_id": model_id,
                "version_id": version_id
            })
            return None
            
    def get_best_version(self,
                        model_id: str,
                        metric: str,
                        higher_better: bool = True) -> Optional[ModelVersion]:
        """Get best performing version.
        
        Args:
            model_id: Model identifier
            metric: Metric to compare
            higher_better: Whether higher is better
            
        Returns:
            Optional[ModelVersion]: Best version
        """
        if model_id not in self.versions:
            return None
            
        versions = self.versions[model_id]
        if not versions:
            return None
            
        # Find best version
        return max(
            versions,
            key=lambda v: v.metrics.get(metric, float('-inf'))
            if higher_better else
            -v.metrics.get(metric, float('inf'))
        )
        
    def compare_versions(self,
                        model_id: str,
                        version_ids: List[str],
                        metrics: List[str]) -> pd.DataFrame:
        """Compare model versions.
        
        Args:
            model_id: Model identifier
            version_ids: Version identifiers
            metrics: Metrics to compare
            
        Returns:
            pd.DataFrame: Version comparison
        """
        import pandas as pd
        
        # Get versions
        versions = []
        for version_id in version_ids:
            version = self.get_version(model_id, version_id)
            if version:
                versions.append(version)
                
        if not versions:
            return pd.DataFrame()
            
        # Create comparison
        data = []
        for version in versions:
            row = {
                "version_id": version.version_id,
                "timestamp": version.timestamp,
                "age_hours": version.age_hours
            }
            
            # Add metrics
            for metric in metrics:
                row[metric] = version.metrics.get(metric)
                
            data.append(row)
            
        return pd.DataFrame(data)
        
    def get_model_lineage(self, model_id: str) -> List[Dict[str, Any]]:
        """Get model version lineage.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List[Dict[str, Any]]: Version lineage
        """
        if model_id not in self.versions:
            return []
            
        lineage = []
        for version in self.versions[model_id]:
            # Get MLflow run
            run = self.mlflow_client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=f"tags.version_id = '{version.version_id}'"
            )
            
            if run:
                run = run[0]
                entry = {
                    "version_id": version.version_id,
                    "timestamp": version.timestamp,
                    "metrics": version.metrics,
                    "metadata": version.metadata,
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_metrics": run.data.metrics,
                    "mlflow_params": run.data.params
                }
                lineage.append(entry)
                
        return lineage
        
    def delete_version(self,
                      model_id: str,
                      version_id: str) -> bool:
        """Delete model version.
        
        Args:
            model_id: Model identifier
            version_id: Version identifier
            
        Returns:
            bool: Whether version was deleted
        """
        if model_id not in self.versions:
            return False
            
        # Find version
        version = None
        for v in self.versions[model_id]:
            if v.version_id == version_id:
                version = v
                break
                
        if not version:
            return False
            
        # Remove version
        self.versions[model_id].remove(version)
        
        # Delete files
        if os.path.exists(version.path):
            os.remove(version.path)
            
        # Save state
        self.save_state()
        
        # Log deletion
        self.log_event(
            "version_deleted",
            {
                "model_id": model_id,
                "version_id": version_id
            }
        )
        
        return True
        
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dict[str, Any]: Registry statistics
        """
        stats = {
            "total_models": len(self.versions),
            "total_versions": sum(
                len(versions)
                for versions in self.versions.values()
            ),
            "models": {}
        }
        
        # Get model stats
        for model_id, versions in self.versions.items():
            model_stats = {
                "versions": len(versions),
                "latest_version": versions[-1].version_id if versions else None,
                "latest_timestamp": versions[-1].timestamp if versions else None,
                "metrics": {}
            }
            
            # Aggregate metrics
            if versions:
                metrics = set()
                for v in versions:
                    metrics.update(v.metrics.keys())
                    
                for metric in metrics:
                    values = [
                        v.metrics[metric]
                        for v in versions
                        if metric in v.metrics
                    ]
                    if values:
                        model_stats["metrics"][metric] = {
                            "mean": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values)
                        }
                        
            stats["models"][model_id] = model_stats
            
        return stats