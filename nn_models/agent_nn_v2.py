# MIGRATED FROM: archive/legacy/nn_models_deprecated/agent_nn_v2.py
# deprecated – moved for cleanup in v1.0.0-beta
"""Neural network for agent task prediction and performance tracking."""
import os
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from utils.logging_util import LoggerMixin
from .model_init import ModelInitializer
from .model_security import ModelSecurityManager

@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    response_time: float
    confidence_score: float
    user_feedback: Optional[float] = None
    task_success: Optional[bool] = None

class AgentNN(LoggerMixin):
    def __init__(self, domain: str):
        """Initialize the neural network.
        
        Args:
            domain: Domain name for the agent
        """
        super().__init__()
        self.domain = domain
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.model_info = None
        self.training_losses: List[float] = []
        self.eval_metrics: List[Dict[str, float]] = []
        
        # Initialize managers
        self.model_manager = ModelInitializer()
        self.security_manager = ModelSecurityManager()
        
        # Model configuration
        self.model_config = {
            "input_size": 768,
            "hidden_size": 256,
            "output_size": 64,
            "extra_layers": 1,
            "output_activation": "tanh",
            "layer_types": ["Linear", "ReLU", "Dropout", "Tanh"]
        }
        
    def load_model(self, path: Optional[str] = None):
        """Load model from file.
        
        Args:
            path: Optional path to model file. If not provided,
                 will use default path for domain
        """
        try:
            # Get model path
            if not path:
                path = os.path.join("models/agent_nn", f"{self.domain}_nn.pt")
                
            # Try to load existing model securely
            model, status = self.security_manager.secure_load(
                path,
                self.model_config,
                device="cpu",
                strict=False
            )
            
            if model is not None and status["status"] == "success":
                self.model = model
                self.model_info = {
                    "model": model,
                    "config": self.model_config,
                    "path": path,
                    "status": "loaded",
                    "security": status
                }
                
                # Create optimizer
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.model_info["config"].get("learning_rate", 0.001),
                    weight_decay=self.model_info["config"].get("weight_decay", 0.01)
                )
                
                # Load metadata if available
                metadata = status.get("metadata")
                if metadata:
                    history = metadata.get("metadata", {})
                    self.training_losses = history.get("training_losses", [])
                    self.eval_metrics = history.get("eval_metrics", [])
                    
                self.log_event(
                    "model_loaded",
                    {
                        "domain": self.domain,
                        "status": "success",
                        "path": path,
                        "training_history": len(self.training_losses),
                        "security_status": status
                    }
                )
                
            else:
                # Create new model
                self.log_event(
                    "creating_new_model",
                    {
                        "domain": self.domain,
                        "reason": status.get("error", "Unknown error")
                    }
                )
                
                # Create model through security manager
                model = self.security_manager._create_model_instance(self.model_config)
                if model is not None:
                    self.model = model
                    self.model_info = {
                        "model": model,
                        "config": self.model_config,
                        "path": path,
                        "status": "new"
                    }
                    
                    # Create optimizer
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(),
                        lr=self.model_config.get("learning_rate", 0.001),
                        weight_decay=self.model_config.get("weight_decay", 0.01)
                    )
                    
                    # Save initial model
                    self.save_model(path)
                    
                else:
                    raise ValueError("Failed to create new model")
                    
        except Exception as e:
            self.log_error(e, {
                "domain": self.domain,
                "operation": "load_model",
                "path": path
            })
            # Use dummy model as fallback
            self.model_info = self.model_manager._create_dummy_model(self.domain)
            self.model = self.model_info["model"]
            self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def save_model(self, path: Optional[str] = None):
        """Save model to file.
        
        Args:
            path: Optional path to save model. If not provided,
                 will use default path for domain
        """
        if not self.model or not self.optimizer:
            return
            
        try:
            # Use default path if none provided
            if not path:
                path = os.path.join("models/agent_nn", f"{self.domain}_nn.pt")
                
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model_config,
                "domain": self.domain,
                "version": "1.0.0"
            }
            
            # Save with weights_only=True
            torch.save(state_dict, path)
            
            # Save metadata separately
            metadata = {
                "domain": self.domain,
                "config": self.model_config,
                "creation_time": datetime.now().isoformat(),
                "last_validated": datetime.now().isoformat(),
                "validation_status": True,
                "trusted": True,
                "training_losses": self.training_losses,
                "eval_metrics": self.eval_metrics,
                "model_stats": {
                    "total_parameters": sum(
                        p.numel() for p in self.model.parameters()
                    ),
                    "trainable_parameters": sum(
                        p.numel() for p in self.model.parameters() if p.requires_grad
                    ),
                    "layer_sizes": [
                        (name, p.size())
                        for name, p in self.model.named_parameters()
                    ]
                }
            }
            
            # Save metadata with security info
            self.security_manager.save_model_metadata(path, metadata)
            
            self.log_event(
                "model_saved",
                {
                    "domain": self.domain,
                    "path": path,
                    "training_history": len(self.training_losses),
                    "model_stats": metadata["model_stats"]
                }
            )
            
        except Exception as e:
            self.log_error(e, {
                "domain": self.domain,
                "operation": "save_model",
                "path": path
            })
        
    def predict_task_features(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict task features from embedding.
        
        Args:
            embedding: Task embedding tensor
            
        Returns:
            torch.Tensor: Predicted task features
        """
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            try:
                return self.model(embedding)
            except Exception as e:
                self.log_error(e, {
                    "domain": self.domain,
                    "operation": "predict_features",
                    "embedding_shape": list(embedding.shape)
                })
                # Return zero features on error
                return torch.zeros(
                    embedding.size(0),
                    self.model_info["config"]["output_size"]
                )
            
    def evaluate_performance(self, metrics: TaskMetrics):
        """Update model based on task performance.
        
        Args:
            metrics: Task execution metrics
        """
        if self.model is None or self.model_info["status"] == "dummy":
            return
            
        try:
            # Convert metrics to tensor
            target = torch.tensor([
                metrics.response_time,
                metrics.confidence_score,
                metrics.user_feedback or 0.0,
                float(metrics.task_success or False)
            ]).unsqueeze(0)
            
            # Update model
            self.optimizer.zero_grad()
            output = self.model(target)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Record metrics
            self.training_losses.append(float(loss))
            self.eval_metrics.append({
                "response_time": metrics.response_time,
                "confidence_score": metrics.confidence_score,
                "user_feedback": metrics.user_feedback,
                "task_success": metrics.task_success
            })
            
            self.log_event(
                "model_updated",
                {
                    "domain": self.domain,
                    "loss": float(loss),
                    "metrics": {
                        "response_time": metrics.response_time,
                        "confidence_score": metrics.confidence_score,
                        "user_feedback": metrics.user_feedback,
                        "task_success": metrics.task_success
                    }
                }
            )
            
        except Exception as e:
            self.log_error(e, {
                "domain": self.domain,
                "operation": "evaluate_performance"
            })
            
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status information.
        
        Returns:
            Dict containing model status
        """
        if not self.model_info:
            return {
                "status": "not_initialized",
                "domain": self.domain
            }
            
        # Get basic status
        status = {
            "status": self.model_info["status"],
            "domain": self.domain,
            "path": self.model_info["path"],
            "config": self.model_info["config"]
        }
        
        # Add security information
        if "security" in self.model_info:
            status["security"] = self.model_info["security"]
            
        # Add model statistics
        if self.model is not None:
            status["model_stats"] = {
                "total_parameters": sum(
                    p.numel() for p in self.model.parameters()
                ),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "layer_sizes": [
                    (name, list(p.size()))
                    for name, p in self.model.named_parameters()
                ]
            }
            
        # Add training history
        status["training"] = {
            "total_updates": len(self.training_losses),
            "total_evaluations": len(self.eval_metrics)
        }
        
        if self.training_losses:
            status["training"].update({
                "avg_loss": float(np.mean(self.training_losses[-100:])),
                "min_loss": float(np.min(self.training_losses)),
                "max_loss": float(np.max(self.training_losses))
            })
            
        if self.eval_metrics:
            recent_metrics = self.eval_metrics[-100:]
            status["performance"] = {
                "avg_response_time": float(np.mean([m["response_time"] for m in recent_metrics])),
                "avg_confidence": float(np.mean([m["confidence_score"] for m in recent_metrics])),
                "success_rate": float(np.mean([
                    1.0 if m["task_success"] else 0.0
                    for m in recent_metrics
                    if m["task_success"] is not None
                ]))
            }
            
        # Get metadata from security manager
        metadata = self.security_manager.load_model_metadata(self.model_info["path"])
        if metadata:
            status["metadata"] = metadata
            
        return status
        
    def verify_model_security(self) -> Dict[str, Any]:
        """Verify model security status.
        
        Returns:
            Dict containing security verification results
        """
        if not self.model or not self.model_info:
            return {
                "status": "error",
                "error": "Model not initialized"
            }
            
        try:
            # Verify model structure
            structure_valid = self.security_manager.validate_model_structure(
                self.model,
                self.model_config
            )
            
            # Get current hash
            current_hash = self.security_manager.compute_model_hash(
                self.model_info["path"]
            )
            
            # Get metadata
            metadata = self.security_manager.load_model_metadata(
                self.model_info["path"]
            )
            
            # Check hash if metadata exists
            hash_valid = True
            if metadata:
                stored_hash = metadata["security"]["model_hash"]
                hash_valid = current_hash == stored_hash
                
            return {
                "status": "verified" if structure_valid and hash_valid else "invalid",
                "structure_valid": structure_valid,
                "hash_valid": hash_valid,
                "current_hash": current_hash,
                "metadata": metadata
            }
            
        except Exception as e:
            self.log_error(e, {
                "operation": "verify_model_security"
            })
            return {
                "status": "error",
                "error": str(e)
            }
