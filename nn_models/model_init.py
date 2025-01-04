"""Neural network model initialization and management."""
import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from pathlib import Path
from utils.logging_util import LoggerMixin

class DefaultModelArchitecture(nn.Module):
    """Default neural network architecture for agent models."""
    
    def __init__(self, input_size: int = 768, hidden_size: int = 256, output_size: int = 64):
        """Initialize the model.
        
        Args:
            input_size: Size of input embeddings
            hidden_size: Size of hidden layers
            output_size: Size of output features
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()  # Normalize outputs to [-1, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output features
        """
        return self.network(x)

class ModelInitializer(LoggerMixin):
    """Initialize and manage neural network models."""
    
    def __init__(self, models_dir: str = "models/agent_nn"):
        """Initialize the model manager.
        
        Args:
            models_dir: Directory to store models
        """
        super().__init__()
        self.models_dir = Path(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default model configuration
        self.default_config = {
            "input_size": 768,  # Size of embeddings
            "hidden_size": 256,
            "output_size": 64,  # Size of feature vectors
            "learning_rate": 0.001,
            "weight_decay": 0.01
        }
        
    def initialize_model(self,
                        domain: str,
                        config: Optional[Dict[str, Any]] = None) -> str:
        """Initialize a new model for a domain.
        
        Args:
            domain: Domain name
            config: Optional model configuration
            
        Returns:
            str: Path to the saved model
        """
        model_path = self.models_dir / f"{domain}_nn.pt"
        
        try:
            # Create model with given or default config
            model_config = config or self.default_config
            model = DefaultModelArchitecture(
                input_size=model_config["input_size"],
                hidden_size=model_config["hidden_size"],
                output_size=model_config["output_size"]
            )
            
            # Initialize weights
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.01)
            model.apply(init_weights)
            
            # Save model
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "domain": domain,
                "version": "1.0.0"
            }, model_path)
            
            self.log_event(
                "model_initialized",
                {
                    "domain": domain,
                    "path": str(model_path),
                    "config": model_config
                }
            )
            
            return str(model_path)
            
        except Exception as e:
            self.log_error(e, {
                "domain": domain,
                "operation": "initialize_model"
            })
            return ""
            
    def load_or_create_model(self,
                            domain: str,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load an existing model or create a new one.
        
        Args:
            domain: Domain name
            config: Optional model configuration
            
        Returns:
            Dict containing model information
        """
        model_path = self.models_dir / f"{domain}_nn.pt"
        
        try:
            if model_path.exists():
                # Try to load existing model
                checkpoint = torch.load(model_path)
                model = DefaultModelArchitecture(
                    input_size=checkpoint["config"]["input_size"],
                    hidden_size=checkpoint["config"]["hidden_size"],
                    output_size=checkpoint["config"]["output_size"]
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                
                self.log_event(
                    "model_loaded",
                    {
                        "domain": domain,
                        "path": str(model_path),
                        "version": checkpoint.get("version", "unknown")
                    }
                )
                
                return {
                    "model": model,
                    "config": checkpoint["config"],
                    "path": str(model_path),
                    "status": "loaded"
                }
            else:
                # Create new model
                new_path = self.initialize_model(domain, config)
                if new_path:
                    return self.load_or_create_model(domain, config)
                else:
                    # Return dummy model if initialization fails
                    return self._create_dummy_model(domain)
                    
        except Exception as e:
            self.log_error(e, {
                "domain": domain,
                "operation": "load_or_create_model"
            })
            return self._create_dummy_model(domain)
            
    def _create_dummy_model(self, domain: str) -> Dict[str, Any]:
        """Create a dummy model that returns zero features.
        
        Args:
            domain: Domain name
            
        Returns:
            Dict containing dummy model information
        """
        class DummyModel(nn.Module):
            def __init__(self, output_size: int = 64):
                super().__init__()
                self.output_size = output_size
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.size(0), self.output_size)
                
        model = DummyModel(self.default_config["output_size"])
        
        self.log_event(
            "dummy_model_created",
            {
                "domain": domain,
                "output_size": self.default_config["output_size"]
            }
        )
        
        return {
            "model": model,
            "config": self.default_config,
            "path": "",
            "status": "dummy"
        }
        
    def get_model_info(self, domain: str) -> Dict[str, Any]:
        """Get information about a model.
        
        Args:
            domain: Domain name
            
        Returns:
            Dict containing model information
        """
        model_path = self.models_dir / f"{domain}_nn.pt"
        
        if not model_path.exists():
            return {
                "exists": False,
                "path": str(model_path),
                "status": "not_found"
            }
            
        try:
            checkpoint = torch.load(model_path)
            return {
                "exists": True,
                "path": str(model_path),
                "config": checkpoint["config"],
                "version": checkpoint.get("version", "unknown"),
                "status": "available"
            }
        except Exception as e:
            self.log_error(e, {
                "domain": domain,
                "operation": "get_model_info"
            })
            return {
                "exists": True,
                "path": str(model_path),
                "status": "error",
                "error": str(e)
            }