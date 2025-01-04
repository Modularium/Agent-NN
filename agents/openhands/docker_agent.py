from typing import Dict, Any, Optional, List, Union
import asyncio
import json
import os
from datetime import datetime
from .base_openhands_agent import OpenHandsAgent

class DockerAgent(OpenHandsAgent):
    """Agent for managing Docker operations."""
    
    def __init__(self,
                 name: str = "docker_agent",
                 api_url: Optional[str] = None,
                 github_token: Optional[str] = None):
        """Initialize Docker agent.
        
        Args:
            name: Agent name
            api_url: OpenHands API URL
            github_token: GitHub token for API access
        """
        super().__init__(name=name, api_url=api_url, github_token=github_token)
        
    async def build_image(self,
                         dockerfile: str,
                         context: Dict[str, str],
                         tag: str) -> Dict[str, Any]:
        """Build Docker image.
        
        Args:
            dockerfile: Dockerfile content
            context: Build context files
            tag: Image tag
            
        Returns:
            Dict[str, Any]: Build results
        """
        await self.initialize()
        
        data = {
            "dockerfile": dockerfile,
            "context": context,
            "tag": tag
        }
        
        async with self.session.post(
            f"{self.api_url}/docker/build",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "image_built",
                    {
                        "tag": tag,
                        "context_size": len(context)
                    }
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to build image: {error}"),
                    {"tag": tag}
                )
                raise Exception(f"Failed to build image: {error}")
                
    async def run_container(self,
                          image: str,
                          command: Optional[str] = None,
                          environment: Optional[Dict[str, str]] = None,
                          ports: Optional[Dict[str, str]] = None,
                          volumes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run Docker container.
        
        Args:
            image: Image name
            command: Optional command to run
            environment: Optional environment variables
            ports: Optional port mappings
            volumes: Optional volume mappings
            
        Returns:
            Dict[str, Any]: Container details
        """
        await self.initialize()
        
        data = {
            "image": image,
            "command": command,
            "environment": environment or {},
            "ports": ports or {},
            "volumes": volumes or {}
        }
        
        async with self.session.post(
            f"{self.api_url}/docker/run",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "container_started",
                    {
                        "image": image,
                        "container_id": result.get("container_id")
                    }
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to run container: {error}"),
                    {"image": image}
                )
                raise Exception(f"Failed to run container: {error}")
                
    async def stop_container(self, container_id: str) -> Dict[str, Any]:
        """Stop Docker container.
        
        Args:
            container_id: Container ID
            
        Returns:
            Dict[str, Any]: Operation result
        """
        await self.initialize()
        
        async with self.session.post(
            f"{self.api_url}/docker/stop/{container_id}"
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "container_stopped",
                    {"container_id": container_id}
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to stop container: {error}"),
                    {"container_id": container_id}
                )
                raise Exception(f"Failed to stop container: {error}")
                
    async def get_container_logs(self,
                               container_id: str,
                               tail: Optional[int] = None) -> str:
        """Get container logs.
        
        Args:
            container_id: Container ID
            tail: Optional number of lines to retrieve
            
        Returns:
            str: Container logs
        """
        await self.initialize()
        
        params = {}
        if tail is not None:
            params["tail"] = str(tail)
            
        async with self.session.get(
            f"{self.api_url}/docker/logs/{container_id}",
            params=params
        ) as response:
            if response.status == 200:
                return await response.text()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get container logs: {error}"),
                    {"container_id": container_id}
                )
                raise Exception(f"Failed to get container logs: {error}")
                
    async def get_container_status(self, container_id: str) -> Dict[str, Any]:
        """Get container status.
        
        Args:
            container_id: Container ID
            
        Returns:
            Dict[str, Any]: Container status
        """
        await self.initialize()
        
        async with self.session.get(
            f"{self.api_url}/docker/status/{container_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get container status: {error}"),
                    {"container_id": container_id}
                )
                raise Exception(f"Failed to get container status: {error}")
                
    async def wait_for_container(self,
                               container_id: str,
                               timeout: float = 300,
                               poll_interval: float = 2) -> Dict[str, Any]:
        """Wait for container to finish.
        
        Args:
            container_id: Container ID
            timeout: Maximum wait time in seconds
            poll_interval: Time between polls in seconds
            
        Returns:
            Dict[str, Any]: Container status
        """
        start_time = datetime.now()
        while True:
            status = await self.get_container_status(container_id)
            if status["state"] not in ["running", "created"]:
                return status
                
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Container {container_id} timed out")
                
            await asyncio.sleep(poll_interval)
            
    async def cleanup_containers(self,
                               label: Optional[str] = None,
                               older_than: Optional[int] = None) -> List[str]:
        """Clean up containers.
        
        Args:
            label: Optional label to filter containers
            older_than: Optional age in seconds
            
        Returns:
            List[str]: Cleaned container IDs
        """
        await self.initialize()
        
        params = {}
        if label:
            params["label"] = label
        if older_than:
            params["older_than"] = str(older_than)
            
        async with self.session.post(
            f"{self.api_url}/docker/cleanup",
            params=params
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "containers_cleaned",
                    {
                        "count": len(result["containers"]),
                        "label": label,
                        "older_than": older_than
                    }
                )
                return result["containers"]
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to cleanup containers: {error}"),
                    {"label": label}
                )
                raise Exception(f"Failed to cleanup containers: {error}")