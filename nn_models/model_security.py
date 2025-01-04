"""Security utilities for model loading and validation."""
import os
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
import torch
from torch import nn
from pathlib import Path
from utils.logging_util import LoggerMixin

class ModelSecurityManager(LoggerMixin):
    """Manager for secure model loading and validation."""
    
    def __init__(self, security_dir: str = "models/security"):
        """Initialize the security manager.
        
        Args:
            security_dir: Directory for security metadata
        """
        super().__init__()
        self.security_dir = Path(security_dir)
        os.makedirs(self.security_dir, exist_ok=True)
        
    def compute_model_hash(self, model_path: str) -> str:
        """Compute SHA-256 hash of model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            str: Hex digest of hash
        """
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def validate_model_structure(self,
                               model: nn.Module,
                               expected_config: Dict[str, Any]) -> bool:
        """Validate model structure matches expected configuration.
        
        Args:
            model: PyTorch model to validate
            expected_config: Expected model configuration
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check input size
            first_layer = next(model.parameters()).size()
            if first_layer[1] != expected_config.get("input_size"):
                self.log_error(
                    ValueError("Input size mismatch"),
                    {
                        "expected": expected_config.get("input_size"),
                        "actual": first_layer[1]
                    }
                )
                return False
                
            # Check output size
            last_layer = list(model.parameters())[-1].size()
            if last_layer[0] != expected_config.get("output_size"):
                self.log_error(
                    ValueError("Output size mismatch"),
                    {
                        "expected": expected_config.get("output_size"),
                        "actual": last_layer[0]
                    }
                )
                return False
                
            # Check layer types
            layer_types = [type(m).__name__ for m in model.modules()]
            expected_types = expected_config.get("layer_types", [])
            if not all(t in layer_types for t in expected_types):
                self.log_error(
                    ValueError("Layer types mismatch"),
                    {
                        "expected": expected_types,
                        "actual": layer_types
                    }
                )
                return False
                
            return True
            
        except Exception as e:
            self.log_error(e, {
                "operation": "validate_model_structure"
            })
            return False
            
    def save_model_metadata(self,
                          model_path: str,
                          metadata: Dict[str, Any]) -> None:
        """Save model metadata including security information.
        
        Args:
            model_path: Path to model file
            metadata: Model metadata
        """
        try:
            # Compute model hash
            model_hash = self.compute_model_hash(model_path)
            
            # Add security information
            security_info = {
                "model_hash": model_hash,
                "creation_time": metadata.get("creation_time"),
                "last_validated": metadata.get("last_validated"),
                "validation_status": metadata.get("validation_status", False),
                "trusted": metadata.get("trusted", False)
            }
            
            # Save metadata
            meta_path = self.security_dir / f"{Path(model_path).stem}_meta.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "security": security_info,
                    "metadata": metadata
                }, f, indent=2)
                
            self.log_event(
                "metadata_saved",
                {
                    "model_path": model_path,
                    "meta_path": str(meta_path)
                }
            )
            
        except Exception as e:
            self.log_error(e, {
                "operation": "save_model_metadata",
                "model_path": model_path
            })
            
    def load_model_metadata(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Load model metadata.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Optional[Dict[str, Any]]: Model metadata if available
        """
        try:
            meta_path = self.security_dir / f"{Path(model_path).stem}_meta.json"
            if not meta_path.exists():
                return None
                
            with open(meta_path) as f:
                return json.load(f)
                
        except Exception as e:
            self.log_error(e, {
                "operation": "load_model_metadata",
                "model_path": model_path
            })
            return None
            
    def secure_load(self,
                   model_path: str,
                   expected_config: Dict[str, Any],
                   device: str = "cpu",
                   strict: bool = True) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
        """Securely load a model with validation.
        
        Args:
            model_path: Path to model file
            expected_config: Expected model configuration
            device: Device to load model on
            strict: Whether to enforce strict validation
            
        Returns:
            Tuple[Optional[nn.Module], Dict[str, Any]]: Loaded model and status
        """
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                return None, {
                    "status": "error",
                    "error": "Model file not found"
                }
                
            # Load metadata
            metadata = self.load_model_metadata(model_path)
            current_hash = self.compute_model_hash(model_path)
            
            # Validate hash if metadata exists
            if metadata:
                stored_hash = metadata["security"]["model_hash"]
                if current_hash != stored_hash:
                    self.log_error(
                        ValueError("Model hash mismatch"),
                        {
                            "expected": stored_hash,
                            "actual": current_hash
                        }
                    )
                    if strict:
                        return None, {
                            "status": "error",
                            "error": "Model hash mismatch"
                        }
                        
            # Load model with weights_only=True
            try:
                state_dict = torch.load(
                    model_path,
                    map_location=device,
                    weights_only=True
                )
            except Exception as e:
                self.log_error(e, {
                    "operation": "load_weights",
                    "model_path": model_path
                })
                return None, {
                    "status": "error",
                    "error": f"Failed to load weights: {str(e)}"
                }
                
            # Create model instance
            model = self._create_model_instance(expected_config)
            if model is None:
                return None, {
                    "status": "error",
                    "error": "Failed to create model instance"
                }
                
            # Load state dict
            try:
                model.load_state_dict(state_dict["model_state_dict"])
            except Exception as e:
                self.log_error(e, {
                    "operation": "load_state_dict",
                    "model_path": model_path
                })
                return None, {
                    "status": "error",
                    "error": f"Failed to load state dict: {str(e)}"
                }
                
            # Validate model structure
            if not self.validate_model_structure(model, expected_config):
                if strict:
                    return None, {
                        "status": "error",
                        "error": "Model structure validation failed"
                    }
                    
            # Update metadata
            status = {
                "status": "success",
                "hash": current_hash,
                "validated": True,
                "metadata": metadata
            }
            
            self.log_event(
                "model_loaded",
                {
                    "model_path": model_path,
                    "status": "success",
                    "hash": current_hash
                }
            )
            
            return model, status
            
        except Exception as e:
            self.log_error(e, {
                "operation": "secure_load",
                "model_path": model_path
            })
            return None, {
                "status": "error",
                "error": str(e)
            }
            
    def _create_model_instance(self, config: Dict[str, Any]) -> Optional[nn.Module]:
        """Create a new model instance from configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Optional[nn.Module]: Created model instance
        """
        try:
            # Create layers based on configuration
            layers = []
            
            # Input layer
            layers.append(nn.Linear(config["input_size"], config["hidden_size"]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            
            # Hidden layers
            if "extra_layers" in config:
                for _ in range(config["extra_layers"]):
                    layers.append(nn.Linear(
                        config["hidden_size"],
                        config["hidden_size"]
                    ))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                    
            # Output layer
            layers.append(nn.Linear(config["hidden_size"], config["output_size"]))
            
            # Add output activation if specified
            if config.get("output_activation") == "tanh":
                layers.append(nn.Tanh())
            elif config.get("output_activation") == "sigmoid":
                layers.append(nn.Sigmoid())
                
            return nn.Sequential(*layers)
            
        except Exception as e:
            self.log_error(e, {
                "operation": "create_model_instance",
                "config": config
            })
            return None