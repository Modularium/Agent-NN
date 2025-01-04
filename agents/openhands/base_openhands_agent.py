from typing import Dict, Any, Optional, List, Union
import aiohttp
import asyncio
import json
import os
from datetime import datetime
from utils.logging_util import LoggerMixin
from agents.worker_agent import WorkerAgent

class OpenHandsAgent(WorkerAgent, LoggerMixin):
    """Base agent for OpenHands integration."""
    
    def __init__(self,
                 name: str,
                 api_url: Optional[str] = None,
                 github_token: Optional[str] = None):
        """Initialize OpenHands agent.
        
        Args:
            name: Agent name
            api_url: OpenHands API URL
            github_token: GitHub token for API access
        """
        super().__init__(name=name)
        self.api_url = api_url or os.getenv("OPENHANDS_API_URL", "http://localhost:8000")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        
        if not self.github_token:
            raise ValueError("GitHub token is required")
            
        # Initialize session
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.github_token}",
                    "Content-Type": "application/json"
                }
            )
            
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def submit_code(self,
                         code: str,
                         language: str,
                         task_description: str,
                         requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Submit code for execution and testing.
        
        Args:
            code: Code to execute
            language: Programming language
            task_description: Description of the task
            requirements: Optional list of requirements
            
        Returns:
            Dict[str, Any]: Submission results
        """
        await self.initialize()
        
        data = {
            "code": code,
            "language": language,
            "task_description": task_description,
            "requirements": requirements or []
        }
        
        async with self.session.post(
            f"{self.api_url}/execute",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                self.log_event(
                    "code_submitted",
                    {
                        "language": language,
                        "code_length": len(code)
                    }
                )
                return result
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to submit code: {error}"),
                    {"language": language}
                )
                raise Exception(f"Failed to submit code: {error}")
                
    async def get_execution_result(self, execution_id: str) -> Dict[str, Any]:
        """Get execution results.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Dict[str, Any]: Execution results
        """
        await self.initialize()
        
        async with self.session.get(
            f"{self.api_url}/executions/{execution_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get execution results: {error}"),
                    {"execution_id": execution_id}
                )
                raise Exception(f"Failed to get execution results: {error}")
                
    async def wait_for_execution(self,
                               execution_id: str,
                               timeout: float = 300,
                               poll_interval: float = 2) -> Dict[str, Any]:
        """Wait for execution completion.
        
        Args:
            execution_id: Execution ID
            timeout: Maximum wait time in seconds
            poll_interval: Time between polls in seconds
            
        Returns:
            Dict[str, Any]: Execution results
        """
        start_time = datetime.now()
        while True:
            result = await self.get_execution_result(execution_id)
            if result["status"] in ["completed", "failed", "error"]:
                return result
                
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(f"Execution {execution_id} timed out")
                
            await asyncio.sleep(poll_interval)
            
    async def submit_and_wait(self,
                            code: str,
                            language: str,
                            task_description: str,
                            requirements: Optional[List[str]] = None,
                            timeout: float = 300) -> Dict[str, Any]:
        """Submit code and wait for results.
        
        Args:
            code: Code to execute
            language: Programming language
            task_description: Description of the task
            requirements: Optional list of requirements
            timeout: Maximum wait time in seconds
            
        Returns:
            Dict[str, Any]: Execution results
        """
        # Submit code
        submission = await self.submit_code(
            code,
            language,
            task_description,
            requirements
        )
        
        # Wait for results
        return await self.wait_for_execution(
            submission["execution_id"],
            timeout=timeout
        )
        
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages.
        
        Returns:
            List[str]: Supported languages
        """
        await self.initialize()
        
        async with self.session.get(
            f"{self.api_url}/languages"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get supported languages: {error}"),
                    {}
                )
                raise Exception(f"Failed to get supported languages: {error}")
                
    async def get_system_status(self) -> Dict[str, Any]:
        """Get OpenHands system status.
        
        Returns:
            Dict[str, Any]: System status
        """
        await self.initialize()
        
        async with self.session.get(
            f"{self.api_url}/status"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to get system status: {error}"),
                    {}
                )
                raise Exception(f"Failed to get system status: {error}")
                
    async def report_issue(self,
                         title: str,
                         description: str,
                         execution_id: Optional[str] = None,
                         logs: Optional[str] = None) -> Dict[str, Any]:
        """Report issue to OpenHands.
        
        Args:
            title: Issue title
            description: Issue description
            execution_id: Optional execution ID
            logs: Optional logs
            
        Returns:
            Dict[str, Any]: Issue details
        """
        await self.initialize()
        
        data = {
            "title": title,
            "description": description,
            "execution_id": execution_id,
            "logs": logs
        }
        
        async with self.session.post(
            f"{self.api_url}/issues",
            json=data
        ) as response:
            if response.status == 201:
                issue = await response.json()
                self.log_event(
                    "issue_reported",
                    {
                        "title": title,
                        "execution_id": execution_id
                    }
                )
                return issue
            else:
                error = await response.text()
                self.log_error(
                    Exception(f"Failed to report issue: {error}"),
                    {"title": title}
                )
                raise Exception(f"Failed to report issue: {error}")