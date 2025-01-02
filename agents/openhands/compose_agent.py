from typing import Dict, Any, Optional, List, Union
import asyncio
import json
import os
from datetime import datetime
from .base_openhands_agent import OpenHandsAgent

class DockerComposeAgent(OpenHandsAgent):
    """Agent for managing Docker Compose operations."""
    
    def __init__(self,
                 name: str = "compose_agent",
                 api_url: Optional[str] = None,
                 github_token: Optional[str] = None):
        """Initialize Docker Compose agent.
        
        Args:
            name: Agent name
            api_url: OpenHands API URL
            github_token: GitHub token for API access
        """
        super().__init__(name=name, api_url=api_url, github_token=github_token)
        
    async def deploy_stack(self,
                          compose_file: str,
                          stack_name: str,
                          environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Deploy Docker Compose stack.
        
        Args:
            compose_file: Docker Compose file content
            stack_name: Stack name
            environment: Optional environment variables
            
        Returns:
            Dict[str, Any]: Deployment results
        """
        await self.initialize()
        
        data = {
            "compose_file": compose_file,
            "stack_name": stack_name,
            "environment": environment or {}
        }
        
        async with self.session.post(
            f"{self.api_url}/compose/deploy",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "stack_deployed",
                    {
                        "stack_name": stack_name,
                        "services": len(result.get("services", []))
                    }
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to deploy stack: {error}"),
                    {"stack_name": stack_name}
                )
                raise Exception(f"Failed to deploy stack: {error}")
                
    async def remove_stack(self, stack_name: str) -> Dict[str, Any]:
        """Remove Docker Compose stack.
        
        Args:
            stack_name: Stack name
            
        Returns:
            Dict[str, Any]: Operation result
        """
        await self.initialize()
        
        async with self.session.delete(
            f"{self.api_url}/compose/stacks/{stack_name}"
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "stack_removed",
                    {"stack_name": stack_name}
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to remove stack: {error}"),
                    {"stack_name": stack_name}
                )
                raise Exception(f"Failed to remove stack: {error}")
                
    async def get_stack_status(self, stack_name: str) -> Dict[str, Any]:
        """Get stack status.
        
        Args:
            stack_name: Stack name
            
        Returns:
            Dict[str, Any]: Stack status
        """
        await self.initialize()
        
        async with self.session.get(
            f"{self.api_url}/compose/stacks/{stack_name}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get stack status: {error}"),
                    {"stack_name": stack_name}
                )
                raise Exception(f"Failed to get stack status: {error}")
                
    async def get_stack_logs(self,
                           stack_name: str,
                           service: Optional[str] = None,
                           tail: Optional[int] = None) -> Dict[str, str]:
        """Get stack logs.
        
        Args:
            stack_name: Stack name
            service: Optional service name
            tail: Optional number of lines
            
        Returns:
            Dict[str, str]: Service logs
        """
        await self.initialize()
        
        params = {}
        if service:
            params["service"] = service
        if tail:
            params["tail"] = str(tail)
            
        async with self.session.get(
            f"{self.api_url}/compose/logs/{stack_name}",
            params=params
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get stack logs: {error}"),
                    {"stack_name": stack_name}
                )
                raise Exception(f"Failed to get stack logs: {error}")
                
    async def scale_service(self,
                          stack_name: str,
                          service: str,
                          replicas: int) -> Dict[str, Any]:
        """Scale stack service.
        
        Args:
            stack_name: Stack name
            service: Service name
            replicas: Number of replicas
            
        Returns:
            Dict[str, Any]: Operation result
        """
        await self.initialize()
        
        data = {
            "service": service,
            "replicas": replicas
        }
        
        async with self.session.post(
            f"{self.api_url}/compose/scale/{stack_name}",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "service_scaled",
                    {
                        "stack_name": stack_name,
                        "service": service,
                        "replicas": replicas
                    }
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to scale service: {error}"),
                    {
                        "stack_name": stack_name,
                        "service": service
                    }
                )
                raise Exception(f"Failed to scale service: {error}")
                
    async def wait_for_stack(self,
                           stack_name: str,
                           timeout: float = 300,
                           poll_interval: float = 2) -> Dict[str, Any]:
        """Wait for stack to be ready.
        
        Args:
            stack_name: Stack name
            timeout: Maximum wait time in seconds
            poll_interval: Time between polls in seconds
            
        Returns:
            Dict[str, Any]: Stack status
        """
        start_time = datetime.now()
        while True:
            status = await self.get_stack_status(stack_name)
            if all(s["state"] == "running" for s in status["services"].values()):
                return status
                
            # Check for failures
            failed = [
                name for name, s in status["services"].items()
                if s["state"] in ["failed", "exited"]
            ]
            if failed:
                raise Exception(f"Services failed: {', '.join(failed)}")
                
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Stack {stack_name} timed out")
                
            await asyncio.sleep(poll_interval)
            
    async def update_stack(self,
                         stack_name: str,
                         compose_file: str,
                         environment: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Update existing stack.
        
        Args:
            stack_name: Stack name
            compose_file: New compose file content
            environment: Optional environment variables
            
        Returns:
            Dict[str, Any]: Update results
        """
        await self.initialize()
        
        data = {
            "compose_file": compose_file,
            "environment": environment or {}
        }
        
        async with self.session.put(
            f"{self.api_url}/compose/stacks/{stack_name}",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "stack_updated",
                    {
                        "stack_name": stack_name,
                        "services": len(result.get("services", []))
                    }
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to update stack: {error}"),
                    {"stack_name": stack_name}
                )
                raise Exception(f"Failed to update stack: {error}")
                
    async def get_stack_metrics(self, stack_name: str) -> Dict[str, Any]:
        """Get stack performance metrics.
        
        Args:
            stack_name: Stack name
            
        Returns:
            Dict[str, Any]: Stack metrics
        """
        await self.initialize()
        
        async with self.session.get(
            f"{self.api_url}/compose/metrics/{stack_name}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get stack metrics: {error}"),
                    {"stack_name": stack_name}
                )
                raise Exception(f"Failed to get stack metrics: {error}")