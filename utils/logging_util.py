import logging
import os
from datetime import datetime
from typing import Optional
import json

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
    
    # Create formatters
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
    
    def log_event(self, event_type: str, data: dict):
        """Log a structured event.
        
        Args:
            event_type: Type of event (e.g., "task_start", "task_complete")
            data: Event data to log
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "class": self.__class__.__name__,
            **data
        }
        self.logger.info(json.dumps(event))
    
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
        
        self.logger.error(json.dumps(error_data))
