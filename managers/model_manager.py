from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoModel, AutoTokenizer
import openai
import os
import json
import time
from datetime import datetime
import mlflow
from utils.logging_util import LoggerMixin

class ModelSource:
    """Model source types."""
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"

class ModelType:
    """Model types."""
    LLM = "llm"
    NN = "nn"
    HYBRID = "hybrid"

class ModelManager(LoggerMixin):
    """Manager for model operations."""
    
    def __init__(self,
                 model_dir: str = "models",
                 cache_dir: str = "cache"):
        """Initialize manager.
        
        Args:
            model_dir: Model directory
            cache_dir: Cache directory
        """
        super().__init__()
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("model_management")
        
        # Initialize model registry
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
        
    def _load_registry(self):
        """Load model registry."""
        registry_path = os.path.join(self.model_dir, "registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                self.models = json.load(f)
                
    def _save_registry(self):
        """Save model registry."""
        registry_path = os.path.join(self.model_dir, "registry.json")
        
        # Convert tensors to lists
        models = {}
        for model_id, model_info in self.models.items():
            models[model_id] = {}
            for key, value in model_info.items():
                if hasattr(value, 'detach'):
                    models[model_id][key] = value.detach().numpy().tolist()
                elif isinstance(value, dict):
                    models[model_id][key] = {
                        k: v.detach().numpy().tolist() if hasattr(v, 'detach') else v
                        for k, v in value.items()
                    }
                else:
                    models[model_id][key] = value
            
        with open(registry_path, "w") as f:
            json.dump(models, f, indent=2)
            
    async def load_model(self,
                        name: str,
                        type: str,
                        source: str,
                        config: Dict[str, Any],
                        version: Optional[str] = None) -> Dict[str, Any]:
        """Load model from source.
        
        Args:
            name: Model name
            type: Model type
            source: Model source
            config: Model configuration
            version: Optional version
            
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            # Start MLflow run
            with mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=f"load_{name}"
            ) as run:
                # Log parameters
                mlflow.log_params({
                    "name": name,
                    "type": type,
                    "source": source,
                    "version": version
                })
                
                # Load model based on source
                if source == ModelSource.LOCAL:
                    model_info = await self._load_local_model(
                        name,
                        type,
                        config
                    )
                elif source == ModelSource.HUGGINGFACE:
                    model_info = await self._load_huggingface_model(
                        name,
                        type,
                        config
                    )
                elif source == ModelSource.OPENAI:
                    model_info = await self._load_openai_model(
                        name,
                        type,
                        config
                    )
                else:
                    raise ValueError(f"Unknown source: {source}")
                    
                # Add to registry
                self.models[name] = {
                    "type": type,
                    "source": source,
                    "config": config,
                    "version": version,
                    "loaded_at": datetime.now().isoformat(),
                    "run_id": run.info.run_id,
                    **model_info
                }
                
                # Save registry
                self._save_registry()
                
                return self.models[name]
                
        except Exception as e:
            self.log_error(e, {
                "name": name,
                "type": type,
                "source": source
            })
            raise
            
    async def _load_local_model(self,
                               name: str,
                               type: str,
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Load local model.
        
        Args:
            name: Model name
            type: Model type
            config: Model configuration
            
        Returns:
            Dict[str, Any]: Model information
        """
        model_path = config.get("path")
        if not model_path:
            raise ValueError("Model path not provided")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        if type == ModelType.NN:
            # Load PyTorch model
            model = torch.load(model_path)
            return {
                "model": model,
                "path": model_path,
                "size": os.path.getsize(model_path)
            }
        elif type == ModelType.LLM:
            # Load language model
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return {
                "model": model,
                "tokenizer": tokenizer,
                "path": model_path,
                "size": os.path.getsize(model_path)
            }
        else:
            raise ValueError(f"Unsupported model type: {type}")
            
    async def _load_huggingface_model(self,
                                    name: str,
                                    type: str,
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Load HuggingFace model.
        
        Args:
            name: Model name
            type: Model type
            config: Model configuration
            
        Returns:
            Dict[str, Any]: Model information
        """
        model_id = config.get("model_id")
        if not model_id:
            raise ValueError("Model ID not provided")
            
        # Set cache directory
        cache_dir = os.path.join(self.cache_dir, "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load model and tokenizer
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            **config.get("model_kwargs", {})
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            **config.get("tokenizer_kwargs", {})
        )
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_id": model_id,
            "cache_dir": cache_dir
        }
        
    async def _load_openai_model(self,
                                name: str,
                                type: str,
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """Load OpenAI model.
        
        Args:
            name: Model name
            type: Model type
            config: Model configuration
            
        Returns:
            Dict[str, Any]: Model information
        """
        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("Model name not provided")
            
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("API key not provided")
            
        # Configure OpenAI
        openai.api_key = api_key
        
        # Test API connection
        response = await openai.Model.aretrive(model_name)
        
        return {
            "model_name": model_name,
            "capabilities": response.get("capabilities", {}),
            "created": response.get("created")
        }
        
    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model information.
        
        Args:
            name: Model name
            
        Returns:
            Optional[Dict[str, Any]]: Model information
        """
        return self.models.get(name)
        
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models.
        
        Returns:
            List[Dict[str, Any]]: Model information
        """
        return [
            {
                "name": name,
                **info
            }
            for name, info in self.models.items()
        ]
        
    def get_model_versions(self, name: str) -> List[str]:
        """Get model versions.
        
        Args:
            name: Model name
            
        Returns:
            List[str]: Model versions
        """
        # Get versions from registry
        versions = []
        if name in self.models:
            versions.append("latest")  # Add latest version
            
            # Add saved versions
            if "versions" in self.models[name]:
                versions.extend(list(self.models[name]["versions"].keys()))
            
        # Get versions from MLflow
        try:
            client = mlflow.tracking.MlflowClient()
            mlflow_versions = client.search_model_versions(f"name='{name}'")
            versions.extend([v.version for v in mlflow_versions])
        except Exception:
            pass
            
        return versions
        
    async def unload_model(self, name: str):
        """Unload model.
        
        Args:
            name: Model name
        """
        if name not in self.models:
            return
            
        try:
            model_info = self.models[name]
            
            # Clean up resources
            if "model" in model_info:
                del model_info["model"]
            if "tokenizer" in model_info:
                del model_info["tokenizer"]
                
            # Remove from registry
            del self.models[name]
            self._save_registry()
            
            # Log unload
            self.log_event(
                "model_unloaded",
                {"name": name}
            )
            
        except Exception as e:
            self.log_error(e, {"name": name})
            raise
            
    def get_model_metrics(self, name: str) -> Dict[str, float]:
        """Get model metrics.
        
        Args:
            name: Model name
            
        Returns:
            Dict[str, float]: Model metrics
        """
        if name not in self.models:
            return {}
            
        model_info = self.models[name]
        return model_info.get("metrics", {})
        
    def save_model_version(self,
                          name: str,
                          version: str,
                          metrics: Dict[str, float]) -> Dict[str, Any]:
        """Save model version.
        
        Args:
            name: Model name
            version: Version name
            metrics: Model metrics
            
        Returns:
            Dict[str, Any]: Version information
        """
        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")
            
        # Create version info
        version_info = {
            "version": version,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        # Add version to model info
        if "versions" not in self.models[name]:
            self.models[name]["versions"] = {}
        self.models[name]["versions"][version] = version_info
        
        # Save registry
        self._save_registry()
        
        return version_info