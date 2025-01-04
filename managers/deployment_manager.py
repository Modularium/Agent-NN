from typing import Dict, Any, Optional, List, Union
import docker
import yaml
import os
import json
from datetime import datetime
import asyncio
from utils.logging_util import LoggerMixin

class DeploymentManager(LoggerMixin):
    """Manager for system deployment and scaling."""
    
    def __init__(self,
                 config_path: str = "config/deployment",
                 docker_compose_path: str = "docker/compose"):
        """Initialize deployment manager.
        
        Args:
            config_path: Path to deployment configs
            docker_compose_path: Path to Docker Compose files
        """
        super().__init__()
        self.config_path = config_path
        self.docker_compose_path = docker_compose_path
        
        # Ensure directories exist
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(docker_compose_path, exist_ok=True)
        
        # Initialize Docker client
        self.docker = docker.from_env()
        
        # Track deployments
        self.deployments: Dict[str, Dict[str, Any]] = {}
        
        # Load existing deployments
        self._load_deployments()
        
    def _load_deployments(self):
        """Load existing deployment configurations."""
        config_file = os.path.join(self.config_path, "deployments.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.deployments = json.load(f)
                
    def _save_deployments(self):
        """Save deployment configurations."""
        config_file = os.path.join(self.config_path, "deployments.json")
        with open(config_file, "w") as f:
            json.dump(self.deployments, f, indent=2)
            
    def create_deployment(self,
                         name: str,
                         components: List[str],
                         config: Dict[str, Any]) -> str:
        """Create new deployment configuration.
        
        Args:
            name: Deployment name
            components: Required components
            config: Deployment configuration
            
        Returns:
            str: Deployment ID
        """
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create deployment config
        self.deployments[deployment_id] = {
            "name": name,
            "components": components,
            "config": config,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Generate Docker Compose file
        compose_config = self._generate_compose_config(
            deployment_id,
            components,
            config
        )
        
        # Save compose file
        compose_file = os.path.join(
            self.docker_compose_path,
            f"{deployment_id}.yml"
        )
        with open(compose_file, "w") as f:
            yaml.dump(compose_config, f)
            
        # Save deployments
        self._save_deployments()
        
        # Log creation
        self.log_event(
            "deployment_created",
            {
                "deployment_id": deployment_id,
                "name": name,
                "components": components
            }
        )
        
        return deployment_id
        
    def _generate_compose_config(self,
                               deployment_id: str,
                               components: List[str],
                               config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Docker Compose configuration.
        
        Args:
            deployment_id: Deployment identifier
            components: Required components
            config: Deployment configuration
            
        Returns:
            Dict[str, Any]: Compose configuration
        """
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {
                "agent_network": {
                    "driver": "bridge"
                }
            }
        }
        
        # Add Redis service
        compose_config["services"]["redis"] = {
            "image": "redis:latest",
            "ports": ["6379:6379"],
            "networks": ["agent_network"],
            "healthcheck": {
                "test": ["CMD", "redis-cli", "ping"],
                "interval": "5s",
                "timeout": "3s",
                "retries": 3
            }
        }
        
        # Add API service
        compose_config["services"]["api"] = {
            "build": {
                "context": ".",
                "dockerfile": "docker/Dockerfile.api"
            },
            "ports": ["8000:8000"],
            "networks": ["agent_network"],
            "depends_on": ["redis"],
            "environment": {
                "REDIS_URL": "redis://redis:6379",
                "LOG_LEVEL": "INFO"
            },
            "deploy": {
                "replicas": config.get("api_replicas", 1),
                "resources": {
                    "limits": {
                        "cpus": "1",
                        "memory": "1G"
                    }
                }
            }
        }
        
        # Add worker services
        for component in components:
            compose_config["services"][component] = {
                "build": {
                    "context": ".",
                    "dockerfile": f"docker/Dockerfile.{component}"
                },
                "networks": ["agent_network"],
                "depends_on": ["redis", "api"],
                "environment": {
                    "REDIS_URL": "redis://redis:6379",
                    "API_URL": "http://api:8000",
                    "LOG_LEVEL": "INFO"
                },
                "deploy": {
                    "replicas": config.get(f"{component}_replicas", 1),
                    "resources": {
                        "limits": {
                            "cpus": "2",
                            "memory": "4G"
                        }
                    }
                }
            }
            
        return compose_config
        
    async def deploy(self, deployment_id: str):
        """Deploy system components.
        
        Args:
            deployment_id: Deployment identifier
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Unknown deployment: {deployment_id}")
            
        deployment = self.deployments[deployment_id]
        compose_file = os.path.join(
            self.docker_compose_path,
            f"{deployment_id}.yml"
        )
        
        try:
            # Update status
            deployment["status"] = "deploying"
            deployment["last_updated"] = datetime.now().isoformat()
            self._save_deployments()
            
            # Deploy stack
            self.docker.compose.up(
                project_name=deployment_id,
                compose_files=[compose_file],
                detach=True
            )
            
            # Update status
            deployment["status"] = "deployed"
            deployment["deployed_at"] = datetime.now().isoformat()
            self._save_deployments()
            
            # Log deployment
            self.log_event(
                "deployment_successful",
                {"deployment_id": deployment_id}
            )
            
        except Exception as e:
            # Update status
            deployment["status"] = "failed"
            deployment["error"] = str(e)
            self._save_deployments()
            
            # Log error
            self.log_error(e, {"deployment_id": deployment_id})
            raise
            
    async def undeploy(self, deployment_id: str):
        """Remove deployed system.
        
        Args:
            deployment_id: Deployment identifier
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Unknown deployment: {deployment_id}")
            
        deployment = self.deployments[deployment_id]
        compose_file = os.path.join(
            self.docker_compose_path,
            f"{deployment_id}.yml"
        )
        
        try:
            # Update status
            deployment["status"] = "removing"
            deployment["last_updated"] = datetime.now().isoformat()
            self._save_deployments()
            
            # Remove stack
            self.docker.compose.down(
                project_name=deployment_id,
                compose_files=[compose_file]
            )
            
            # Update status
            deployment["status"] = "removed"
            deployment["removed_at"] = datetime.now().isoformat()
            self._save_deployments()
            
            # Log removal
            self.log_event(
                "deployment_removed",
                {"deployment_id": deployment_id}
            )
            
        except Exception as e:
            # Update status
            deployment["status"] = "remove_failed"
            deployment["error"] = str(e)
            self._save_deployments()
            
            # Log error
            self.log_error(e, {"deployment_id": deployment_id})
            raise
            
    async def scale_component(self,
                            deployment_id: str,
                            component: str,
                            replicas: int):
        """Scale deployment component.
        
        Args:
            deployment_id: Deployment identifier
            component: Component to scale
            replicas: Number of replicas
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Unknown deployment: {deployment_id}")
            
        deployment = self.deployments[deployment_id]
        if component not in deployment["components"]:
            raise ValueError(f"Unknown component: {component}")
            
        try:
            # Scale service
            self.docker.compose.scale(
                project_name=deployment_id,
                service_scale={component: replicas}
            )
            
            # Update configuration
            deployment["config"][f"{component}_replicas"] = replicas
            deployment["last_updated"] = datetime.now().isoformat()
            self._save_deployments()
            
            # Log scaling
            self.log_event(
                "component_scaled",
                {
                    "deployment_id": deployment_id,
                    "component": component,
                    "replicas": replicas
                }
            )
            
        except Exception as e:
            # Log error
            self.log_error(e, {
                "deployment_id": deployment_id,
                "component": component
            })
            raise
            
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Dict[str, Any]: Deployment status
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Unknown deployment: {deployment_id}")
            
        deployment = self.deployments[deployment_id]
        
        try:
            # Get service status
            services = self.docker.compose.ps(
                project_name=deployment_id,
                services=True
            )
            
            status = {
                "deployment_id": deployment_id,
                "name": deployment["name"],
                "status": deployment["status"],
                "components": {}
            }
            
            # Add component status
            for service in services:
                status["components"][service.name] = {
                    "state": service.state,
                    "status": service.status,
                    "replicas": len([
                        s for s in services
                        if s.name == service.name
                    ])
                }
                
            return status
            
        except Exception as e:
            # Log error
            self.log_error(e, {"deployment_id": deployment_id})
            
            # Return basic status
            return {
                "deployment_id": deployment_id,
                "name": deployment["name"],
                "status": deployment["status"],
                "error": str(e)
            }
            
    def get_deployment_logs(self,
                          deployment_id: str,
                          component: Optional[str] = None) -> Dict[str, str]:
        """Get deployment logs.
        
        Args:
            deployment_id: Deployment identifier
            component: Optional component name
            
        Returns:
            Dict[str, str]: Component logs
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Unknown deployment: {deployment_id}")
            
        try:
            # Get services
            services = self.docker.compose.ps(
                project_name=deployment_id,
                services=True
            )
            
            logs = {}
            for service in services:
                if component is None or service.name == component:
                    logs[service.name] = service.logs().decode()
                    
            return logs
            
        except Exception as e:
            # Log error
            self.log_error(e, {
                "deployment_id": deployment_id,
                "component": component
            })
            raise
            
    def get_deployment_metrics(self,
                             deployment_id: str) -> Dict[str, Dict[str, float]]:
        """Get deployment performance metrics.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Dict[str, Dict[str, float]]: Component metrics
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Unknown deployment: {deployment_id}")
            
        try:
            # Get services
            services = self.docker.compose.ps(
                project_name=deployment_id,
                services=True
            )
            
            metrics = {}
            for service in services:
                stats = service.stats(stream=False)
                
                # Calculate CPU usage
                cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                           stats["precpu_stats"]["cpu_usage"]["total_usage"]
                system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                             stats["precpu_stats"]["system_cpu_usage"]
                cpu_percent = (cpu_delta / system_delta) * 100.0
                
                # Calculate memory usage
                memory_usage = stats["memory_stats"]["usage"]
                memory_limit = stats["memory_stats"]["limit"]
                memory_percent = (memory_usage / memory_limit) * 100.0
                
                metrics[service.name] = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_usage": memory_usage,
                    "memory_limit": memory_limit
                }
                
            return metrics
            
        except Exception as e:
            # Log error
            self.log_error(e, {"deployment_id": deployment_id})
            raise