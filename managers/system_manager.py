from typing import Dict, Any, Optional, List, Union
import os
import json
import shutil
import tarfile
import psutil
import asyncio
from datetime import datetime
import mlflow
from utils.logging_util import LoggerMixin

class SystemConfig:
    """System configuration."""
    def __init__(self,
                 max_concurrent_tasks: int = 10,
                 task_timeout: int = 300,
                 cache_size: int = 1024,  # MB
                 log_level: str = "INFO"):
        """Initialize configuration.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            task_timeout: Task timeout in seconds
            cache_size: Cache size in MB
            log_level: Logging level
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        self.cache_size = cache_size
        self.log_level = log_level
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "task_timeout": self.task_timeout,
            "cache_size": self.cache_size,
            "log_level": self.log_level
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            SystemConfig: Configuration object
        """
        return cls(**data)

class SystemManager(LoggerMixin):
    """Manager for system operations."""
    
    def __init__(self,
                 data_dir: str = "data",
                 backup_dir: str = "backups",
                 config_file: str = "config/system.json"):
        """Initialize manager.
        
        Args:
            data_dir: Data directory
            backup_dir: Backup directory
            config_file: Configuration file
        """
        super().__init__()
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        self.config_file = config_file
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("system_management")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize semaphore
        self.task_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_tasks
        )
        
    def _load_config(self) -> SystemConfig:
        """Load system configuration.
        
        Returns:
            SystemConfig: System configuration
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                return SystemConfig.from_dict(json.load(f))
        return SystemConfig()
        
    def _save_config(self):
        """Save system configuration."""
        with open(self.config_file, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
            
    async def update_config(self,
                          config: Dict[str, Any]) -> SystemConfig:
        """Update system configuration.
        
        Args:
            config: Configuration updates
            
        Returns:
            SystemConfig: Updated configuration
        """
        try:
            # Update configuration
            new_config = SystemConfig.from_dict({
                **self.config.to_dict(),
                **config
            })
            
            # Update semaphore if needed
            if (new_config.max_concurrent_tasks !=
                self.config.max_concurrent_tasks):
                self.task_semaphore = asyncio.Semaphore(
                    new_config.max_concurrent_tasks
                )
                
            # Save configuration
            self.config = new_config
            self._save_config()
            
            # Log update
            self.log_event(
                "config_updated",
                {"config": config}
            )
            
            return self.config
            
        except Exception as e:
            self.log_error(e, {"config": config})
            raise
            
    async def create_backup(self,
                          include_models: bool = True,
                          include_data: bool = True,
                          max_backups: int = 5) -> Dict[str, Any]:
        """Create system backup.
        
        Args:
            include_models: Include model files
            include_data: Include knowledge base data
            max_backups: Maximum number of backups
            
        Returns:
            Dict[str, Any]: Backup information
        """
        try:
            # Generate backup ID
            backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, backup_id)
            os.makedirs(backup_path)
            
            # Create archive
            archive_path = f"{backup_path}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                # Add configuration
                tar.add(self.config_file, arcname="config/system.json")
                
                # Add models
                if include_models:
                    model_dir = "models"
                    if os.path.exists(model_dir):
                        tar.add(model_dir, arcname="models")
                        
                # Add data
                if include_data:
                    if os.path.exists(self.data_dir):
                        tar.add(self.data_dir, arcname="data")
                        
            # Get backup size
            size = os.path.getsize(archive_path)
            
            # Remove old backups
            backups = sorted([
                f for f in os.listdir(self.backup_dir)
                if f.endswith(".tar.gz")
            ])
            while len(backups) > max_backups:
                old_backup = os.path.join(self.backup_dir, backups.pop(0))
                os.remove(old_backup)
                
            # Log backup
            self.log_event(
                "backup_created",
                {
                    "backup_id": backup_id,
                    "size": size,
                    "include_models": include_models,
                    "include_data": include_data
                }
            )
            
            return {
                "backup_id": backup_id,
                "path": archive_path,
                "size": size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(e)
            raise
            
    async def restore_backup(self, backup_id: str):
        """Restore from backup.
        
        Args:
            backup_id: Backup identifier
        """
        try:
            # Find backup
            archive_path = os.path.join(self.backup_dir, f"{backup_id}.tar.gz")
            if not os.path.exists(archive_path):
                raise ValueError(f"Backup not found: {backup_id}")
                
            # Create temporary directory
            temp_dir = os.path.join(self.backup_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract archive
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(temp_dir)
                
            # Restore configuration
            config_path = os.path.join(temp_dir, "config/system.json")
            if os.path.exists(config_path):
                shutil.copy2(config_path, self.config_file)
                self.config = self._load_config()
                
            # Restore models
            model_dir = os.path.join(temp_dir, "models")
            if os.path.exists(model_dir):
                if os.path.exists("models"):
                    shutil.rmtree("models")
                shutil.copytree(model_dir, "models")
                
            # Restore data
            data_dir = os.path.join(temp_dir, "data")
            if os.path.exists(data_dir):
                if os.path.exists(self.data_dir):
                    shutil.rmtree(self.data_dir)
                shutil.copytree(data_dir, self.data_dir)
                
            # Clean up
            shutil.rmtree(temp_dir)
            
            # Log restoration
            self.log_event(
                "backup_restored",
                {"backup_id": backup_id}
            )
            
        except Exception as e:
            self.log_error(e, {"backup_id": backup_id})
            raise
            
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics.
        
        Returns:
            Dict[str, float]: System metrics
        """
        try:
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Get memory metrics
            memory = psutil.virtual_memory()
            
            # Get disk metrics
            disk = psutil.disk_usage("/")
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "memory_percent": memory.percent,
                "memory_available": memory.available / (1024 * 1024),  # MB
                "disk_percent": disk.percent,
                "disk_free": disk.free / (1024 * 1024 * 1024)  # GB
            }
            
        except Exception as e:
            self.log_error(e)
            raise
            
    def get_task_stats(self) -> Dict[str, int]:
        """Get task statistics.
        
        Returns:
            Dict[str, int]: Task statistics
        """
        return {
            "max_tasks": self.config.max_concurrent_tasks,
            "active_tasks": (
                self.config.max_concurrent_tasks -
                self.task_semaphore._value
            )
        }
        
    async def cleanup(self):
        """Clean up system resources."""
        try:
            # Clean temporary files
            temp_dir = os.path.join(self.backup_dir, "temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
            # Log cleanup
            self.log_event("system_cleanup", {})
            
        except Exception as e:
            self.log_error(e)
            raise