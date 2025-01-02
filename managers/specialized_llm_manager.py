from typing import Dict, Any, Optional, List, Union
import torch
from datetime import datetime
import json
import os
from utils.logging_util import LoggerMixin
from nn_models.agent_nn import TaskMetrics
from config.llm_config import OPENAI_CONFIG

class SpecializedLLMManager(LoggerMixin):
    """Manager for domain-specific language models."""
    
    def __init__(self,
                 models_path: str = "models/specialized",
                 cache_path: str = "cache/llm"):
        """Initialize specialized LLM manager.
        
        Args:
            models_path: Path to model files
            cache_path: Path to cache directory
        """
        super().__init__()
        self.models_path = models_path
        self.cache_path = cache_path
        
        # Ensure directories exist
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(cache_path, exist_ok=True)
        
        # Track model performance
        self.model_metrics: Dict[str, List[TaskMetrics]] = {}
        
        # Load model configurations
        self.model_configs = self._load_model_configs()
        
    def _load_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load model configurations.
        
        Returns:
            Dict[str, Dict[str, Any]]: Model configurations
        """
        config_path = os.path.join(self.models_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return {}
        
    def _save_model_configs(self):
        """Save model configurations."""
        config_path = os.path.join(self.models_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.model_configs, f, indent=2)
            
    def add_model(self,
                 model_id: str,
                 domain: str,
                 base_model: str,
                 description: str,
                 parameters: Optional[Dict[str, Any]] = None):
        """Add new specialized model.
        
        Args:
            model_id: Model identifier
            domain: Target domain
            base_model: Base model name
            description: Model description
            parameters: Optional model parameters
        """
        self.model_configs[model_id] = {
            "domain": domain,
            "base_model": base_model,
            "description": description,
            "parameters": parameters or {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "training_status": "initialized"
        }
        
        self._save_model_configs()
        
        # Log addition
        self.log_event(
            "model_added",
            {
                "model_id": model_id,
                "domain": domain,
                "base_model": base_model
            }
        )
        
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dict[str, Any]: Model information
        """
        if model_id not in self.model_configs:
            raise ValueError(f"Unknown model: {model_id}")
            
        info = self.model_configs[model_id].copy()
        
        # Add performance metrics if available
        if model_id in self.model_metrics:
            metrics = self.model_metrics[model_id]
            if metrics:
                info["metrics"] = {
                    "total_tasks": len(metrics),
                    "avg_response_time": sum(m.response_time for m in metrics) / len(metrics),
                    "avg_confidence": sum(m.confidence_score for m in metrics) / len(metrics),
                    "success_rate": sum(1 for m in metrics if m.task_success) / len(metrics)
                }
                
        return info
        
    def get_domain_models(self, domain: str) -> List[str]:
        """Get models for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List[str]: Model identifiers
        """
        return [
            model_id
            for model_id, config in self.model_configs.items()
            if config["domain"] == domain
        ]
        
    def update_model_status(self,
                          model_id: str,
                          status: str,
                          metadata: Optional[Dict[str, Any]] = None):
        """Update model training status.
        
        Args:
            model_id: Model identifier
            status: New status
            metadata: Optional status metadata
        """
        if model_id not in self.model_configs:
            raise ValueError(f"Unknown model: {model_id}")
            
        self.model_configs[model_id]["training_status"] = status
        self.model_configs[model_id]["last_updated"] = datetime.now().isoformat()
        
        if metadata:
            self.model_configs[model_id]["status_metadata"] = metadata
            
        self._save_model_configs()
        
        # Log update
        self.log_event(
            "model_status_updated",
            {
                "model_id": model_id,
                "status": status,
                "metadata": metadata
            }
        )
        
    def update_model_metrics(self,
                           model_id: str,
                           metrics: TaskMetrics):
        """Update model performance metrics.
        
        Args:
            model_id: Model identifier
            metrics: Task metrics
        """
        if model_id not in self.model_configs:
            raise ValueError(f"Unknown model: {model_id}")
            
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = []
            
        self.model_metrics[model_id].append(metrics)
        
        # Log update
        self.log_event(
            "model_metrics_updated",
            {
                "model_id": model_id,
                "metrics": {
                    "response_time": metrics.response_time,
                    "confidence": metrics.confidence_score,
                    "success": metrics.task_success
                }
            }
        )
        
    def get_best_model(self,
                      domain: str,
                      task_description: str) -> Optional[str]:
        """Get best model for task.
        
        Args:
            domain: Task domain
            task_description: Task description
            
        Returns:
            Optional[str]: Best model identifier
        """
        # Get domain models
        models = self.get_domain_models(domain)
        if not models:
            return None
            
        # Calculate scores
        scores = []
        for model_id in models:
            if model_id in self.model_metrics:
                metrics = self.model_metrics[model_id]
                if metrics:
                    # Calculate score based on metrics
                    success_rate = sum(1 for m in metrics if m.task_success) / len(metrics)
                    avg_confidence = sum(m.confidence_score for m in metrics) / len(metrics)
                    score = 0.7 * success_rate + 0.3 * avg_confidence
                    scores.append((model_id, score))
                    
        if not scores:
            # Return first available model if no metrics
            return models[0]
            
        # Return model with highest score
        return max(scores, key=lambda x: x[1])[0]
        
    def remove_model(self, model_id: str):
        """Remove specialized model.
        
        Args:
            model_id: Model identifier
        """
        if model_id not in self.model_configs:
            raise ValueError(f"Unknown model: {model_id}")
            
        # Remove configuration
        del self.model_configs[model_id]
        self._save_model_configs()
        
        # Remove metrics
        if model_id in self.model_metrics:
            del self.model_metrics[model_id]
            
        # Remove model files
        model_path = os.path.join(self.models_path, model_id)
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Log removal
        self.log_event(
            "model_removed",
            {"model_id": model_id}
        )
        
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all models.
        
        Returns:
            Dict[str, Dict[str, Any]]: Model statistics
        """
        stats = {}
        for model_id in self.model_configs:
            info = self.get_model_info(model_id)
            stats[model_id] = {
                "domain": info["domain"],
                "status": info["training_status"],
                "last_updated": info["last_updated"]
            }
            if "metrics" in info:
                stats[model_id]["metrics"] = info["metrics"]
                
        return stats