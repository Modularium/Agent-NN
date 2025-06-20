import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import json
import mlflow
import numpy as np
import torch
from enum import Enum
from dataclasses import asdict, is_dataclass

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dataclasses and special types."""
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record, cls=CustomJSONEncoder)

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    use_json = os.getenv("LOG_JSON", "false").lower() == "true"
    if use_json:
        console_formatter = JsonFormatter()
        file_formatter = JsonFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create main application logger
app_logger = setup_logger(
    'smolit_llm',
    log_file='logs/app.log',
    level=logging.INFO
)

# Create specialized loggers
agent_logger = setup_logger(
    'smolit_llm.agents',
    log_file='logs/agents.log',
    level=logging.INFO
)

model_logger = setup_logger(
    'smolit_llm.models',
    log_file='logs/models.log',
    level=logging.INFO
)

class LoggerMixin:
    """Mixin to add logging capabilities to a class."""
    
    def __init__(self):
        self.logger = setup_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            log_file=f"logs/{self.__class__.__name__.lower()}.log",
            level=logging.INFO
        )
        
        # Initialize MLflow if not already done
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=f"{self.__class__.__name__}_run")
        else:
            try:
                mlflow.start_run(run_name=f"{self.__class__.__name__}_run", nested=True)
            except Exception:
                pass  # Ignore MLflow errors in tests
    
    def log_event(self, event_type: str, data: dict, metrics: Optional[Dict[str, float]] = None):
        """Log a structured event.
        
        Args:
            event_type: Type of event (e.g., "task_start", "task_complete")
            data: Event data to log
            metrics: Optional numerical metrics to track
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "class": self.__class__.__name__,
            **data
        }
        self.logger.info(json.dumps(event, cls=CustomJSONEncoder))
        
        # Log metrics to MLflow
        if metrics:
            try:
                mlflow.log_metrics(metrics)
            except Exception:
                pass  # Ignore MLflow errors in tests
            
        # Log event data as tags
        flat_data = self._flatten_dict(data, prefix='event')
        try:
            mlflow.set_tags(flat_data)
        except Exception:
            pass  # Ignore MLflow errors in tests
    
    def log_error(self, error: Exception, context: Optional[dict] = None):
        """Log an error with context.
        
        Args:
            error: The exception to log
            context: Optional context data
        """
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "class": self.__class__.__name__
        }
        if context:
            error_data["context"] = context
        
        self.logger.error(json.dumps(error_data, cls=CustomJSONEncoder))
        
        # Log error to MLflow
        try:
            mlflow.set_tag("error_type", error.__class__.__name__)
            mlflow.set_tag("error_message", str(error))
            if context:
                flat_context = self._flatten_dict(context, prefix='error_context')
                mlflow.set_tags(flat_context)
        except Exception:
            pass  # Ignore MLflow errors in tests
            
    def log_model_performance(self,
                            model_name: str,
                            metrics: Dict[str, float],
                            metadata: Optional[Dict[str, Any]] = None):
        """Log model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics
            metadata: Additional metadata about the model
        """
        event_data = {
            "model_name": model_name,
            "metrics": metrics,
            **(metadata or {})
        }
        try:
            self.log_event("model_performance", event_data, metrics=metrics)
        except Exception:
            pass  # Ignore MLflow errors in tests
        
    def _flatten_dict(self,
                     d: Dict[str, Any],
                     prefix: str = '',
                     separator: str = '.') -> Dict[str, str]:
        """Flatten a nested dictionary for MLflow tags.
        
        Args:
            d: Dictionary to flatten
            prefix: Prefix for flattened keys
            separator: Separator between nested keys
            
        Returns:
            Dict[str, str]: Flattened dictionary
        """
        items: List[Tuple[str, str]] = []
        
        for k, v in d.items():
            new_key = f"{prefix}{separator}{k}" if prefix else k
            
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(v, new_key, separator).items()
                )
            else:
                items.append((new_key, str(v)))
                
        return dict(items)
        
    def end_run(self):
        """End the current MLflow run."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass  # Ignore MLflow errors in tests
