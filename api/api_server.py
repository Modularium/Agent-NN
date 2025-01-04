from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import json
from datetime import datetime, timedelta
import mlflow
from managers.agent_manager import AgentManager
from managers.monitoring_system import MonitoringSystem
from managers.ab_testing import ABTestingManager
from managers.security_manager import SecurityManager
from utils.logging_util import LoggerMixin

# API Models
class TaskRequest(BaseModel):
    """Task execution request."""
    description: str = Field(..., description="Task description")
    domain: Optional[str] = Field(None, description="Optional domain hint")
    priority: int = Field(1, description="Task priority (1-10)")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context")

class TaskResponse(BaseModel):
    """Task execution response."""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    agent_id: str = Field(..., description="Agent identifier")
    execution_time: float = Field(..., description="Execution time in seconds")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")

class AgentConfig(BaseModel):
    """Agent configuration."""
    name: str = Field(..., description="Agent name")
    domain: str = Field(..., description="Agent domain")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    config: Dict[str, Any] = Field(..., description="Agent configuration")

class SystemMetrics(BaseModel):
    """System metrics."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    active_agents: int = Field(..., description="Number of active agents")
    task_queue_size: int = Field(..., description="Task queue size")
    avg_response_time: float = Field(..., description="Average response time")

class APIServer(LoggerMixin):
    """FastAPI server for Smolit LLM-NN."""
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8000):
        """Initialize API server.
        
        Args:
            host: Server host
            port: Server port
        """
        super().__init__()
        self.host = host
        self.port = port
        
        # Initialize FastAPI
        self.app = FastAPI(
            title="Smolit LLM-NN API",
            description="API for multi-agent neural network system",
            version="2.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Initialize managers
        self.agent_manager = AgentManager()
        self.monitoring = MonitoringSystem()
        self.ab_testing = ABTestingManager()
        self.security = SecurityManager()
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("api_server")
        
        # Add routes
        self._add_routes()
        
    def _add_routes(self):
        """Add API routes."""
        # Authentication
        oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
        @self.app.post("/token")
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            """Login and get access token."""
            # Validate credentials (implement your auth logic)
            if not self._validate_credentials(
                form_data.username,
                form_data.password
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
                
            # Generate token
            token = self.security.generate_token(
                form_data.username,
                ["user"]
            )
            
            return {"access_token": token, "token_type": "bearer"}
            
        # Task Management
        @self.app.post(
            "/tasks",
            response_model=TaskResponse,
            dependencies=[Security(oauth2_scheme)]
        )
        async def create_task(task: TaskRequest):
            """Create and execute task."""
            try:
                # Select agent
                agent = self.agent_manager.select_agent(task.description)
                
                # Execute task
                start_time = datetime.now()
                result = await agent.execute_task(
                    task.description,
                    task.context
                )
                duration = (datetime.now() - start_time).total_seconds()
                
                # Create response
                response = TaskResponse(
                    task_id=result["task_id"],
                    status="completed",
                    result=result["output"],
                    agent_id=agent.id,
                    execution_time=duration,
                    metrics=result["metrics"]
                )
                
                # Log execution
                with mlflow.start_run():
                    mlflow.log_metrics(result["metrics"])
                    mlflow.log_params({
                        "agent_id": agent.id,
                        "duration": duration
                    })
                    
                return response
                
            except Exception as e:
                self.log_error(e, {"task": task.dict()})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
        @self.app.get(
            "/tasks/{task_id}",
            response_model=TaskResponse,
            dependencies=[Security(oauth2_scheme)]
        )
        async def get_task(task_id: str):
            """Get task status and result."""
            try:
                result = await self.agent_manager.get_task_result(task_id)
                if not result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Task not found"
                    )
                return result
            except Exception as e:
                self.log_error(e, {"task_id": task_id})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
        # Agent Management
        @self.app.post(
            "/agents",
            response_model=AgentConfig,
            dependencies=[Security(oauth2_scheme)]
        )
        async def create_agent(config: AgentConfig):
            """Create new agent."""
            try:
                agent = await self.agent_manager.create_new_agent(
                    config.name,
                    config.domain,
                    config.capabilities,
                    config.config
                )
                return agent.get_config()
            except Exception as e:
                self.log_error(e, {"config": config.dict()})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
        @self.app.get(
            "/agents",
            response_model=List[AgentConfig],
            dependencies=[Security(oauth2_scheme)]
        )
        async def list_agents():
            """List all agents."""
            try:
                agents = self.agent_manager.get_all_agents()
                return [agent.get_config() for agent in agents]
            except Exception as e:
                self.log_error(e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
        # Monitoring
        @self.app.get(
            "/metrics",
            response_model=SystemMetrics,
            dependencies=[Security(oauth2_scheme)]
        )
        async def get_metrics():
            """Get system metrics."""
            try:
                metrics = self.monitoring.get_all_statistics()
                return SystemMetrics(
                    cpu_usage=metrics["cpu_usage"]["mean"],
                    memory_usage=metrics["memory_usage"]["mean"],
                    active_agents=len(self.agent_manager.get_all_agents()),
                    task_queue_size=self.agent_manager.get_queue_size(),
                    avg_response_time=metrics["response_time"]["mean"]
                )
            except Exception as e:
                self.log_error(e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
        @self.app.get(
            "/metrics/{metric_name}",
            response_model=Dict[str, float],
            dependencies=[Security(oauth2_scheme)]
        )
        async def get_metric(metric_name: str):
            """Get specific metric statistics."""
            try:
                return self.monitoring.get_metric_statistics(metric_name)
            except Exception as e:
                self.log_error(e, {"metric": metric_name})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
        # A/B Testing
        @self.app.post(
            "/tests",
            dependencies=[Security(oauth2_scheme)]
        )
        async def create_test(config: Dict[str, Any]):
            """Create new A/B test."""
            try:
                test = self.ab_testing.create_test(**config)
                return {"test_id": test.test_id}
            except Exception as e:
                self.log_error(e, {"config": config})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
        @self.app.get(
            "/tests/{test_id}",
            dependencies=[Security(oauth2_scheme)]
        )
        async def get_test_results(test_id: str):
            """Get A/B test results."""
            try:
                return self.ab_testing.get_test_results(test_id)
            except Exception as e:
                self.log_error(e, {"test_id": test_id})
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )
                
    def _validate_credentials(self,
                            username: str,
                            password: str) -> bool:
        """Validate user credentials.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            bool: Whether credentials are valid
        """
        # Implement your authentication logic
        return True  # Placeholder
        
    async def start(self):
        """Start API server."""
        # Start monitoring
        self.monitoring.start()
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def stop(self):
        """Stop API server."""
        # Stop monitoring
        self.monitoring.stop()
        
        # Stop server
        # Note: uvicorn doesn't provide a clean way to stop programmatically
        pass

if __name__ == "__main__":
    server = APIServer()
    asyncio.run(server.start())