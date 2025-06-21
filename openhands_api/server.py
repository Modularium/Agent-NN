from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import asyncio
import docker
import aioredis
import jwt
import os
from utils.api_utils import api_route
from datetime import datetime, timedelta
from utils.logging_util import LoggerMixin

# Models
class CodeExecution(BaseModel):
    code: str
    language: str
    task_description: str
    requirements: Optional[List[str]] = Field(default_factory=list)

class DockerBuild(BaseModel):
    dockerfile: str
    context: Dict[str, str]
    tag: str

class DockerRun(BaseModel):
    image: str
    command: Optional[str] = None
    environment: Dict[str, str] = Field(default_factory=dict)
    ports: Dict[str, str] = Field(default_factory=dict)
    volumes: Dict[str, str] = Field(default_factory=dict)

class ComposeDeployment(BaseModel):
    compose_file: str
    stack_name: str
    environment: Dict[str, str] = Field(default_factory=dict)

class ServiceScale(BaseModel):
    service: str
    replicas: int

# API Server
class OpenHandsAPI(FastAPI, LoggerMixin):
    """OpenHands API server."""
    
    def __init__(self):
        """Initialize API server."""
        super().__init__(title="OpenHands API")
        LoggerMixin.__init__(self)
        
        # Security
        self.security = HTTPBearer()
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY environment variable is required")
            
        # Docker client
        self.docker = docker.from_env()
        
        # Redis for task queue
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost")
        self.redis = None
        
        # Add CORS middleware
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Register routes
        self._register_routes()
        
    async def initialize(self):
        """Initialize connections."""
        self.redis = await aioredis.from_url(self.redis_url)
        
    async def cleanup(self):
        """Clean up resources."""
        if self.redis:
            await self.redis.close()
            
    def _register_routes(self):
        """Register API routes."""
        # Code execution routes
        self.add_api_route(
            "/execute",
            self.execute_code,
            methods=["POST"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        self.add_api_route(
            "/executions/{execution_id}",
            self.get_execution,
            methods=["GET"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        
        # Docker routes
        self.add_api_route(
            "/docker/build",
            self.build_image,
            methods=["POST"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        self.add_api_route(
            "/docker/run",
            self.run_container,
            methods=["POST"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        self.add_api_route(
            "/docker/stop/{container_id}",
            self.stop_container,
            methods=["POST"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        self.add_api_route(
            "/docker/logs/{container_id}",
            self.get_container_logs,
            methods=["GET"],
            response_model=str,
            dependencies=[Depends(self.verify_token)]
        )
        
        # Docker Compose routes
        self.add_api_route(
            "/compose/deploy",
            self.deploy_stack,
            methods=["POST"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        self.add_api_route(
            "/compose/stacks/{stack_name}",
            self.get_stack_status,
            methods=["GET"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        self.add_api_route(
            "/compose/scale/{stack_name}",
            self.scale_service,
            methods=["POST"],
            response_model=Dict[str, Any],
            dependencies=[Depends(self.verify_token)]
        )
        
        # System routes
        self.add_api_route(
            "/status",
            self.get_system_status,
            methods=["GET"],
            response_model=Dict[str, Any]
        )
        self.add_api_route(
            "/languages",
            self.get_supported_languages,
            methods=["GET"],
            response_model=List[str]
        )
        
    async def verify_token(self,
                          credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> bool:
        """Verify JWT token.
        
        Args:
            credentials: Bearer token credentials
            
        Returns:
            bool: Whether token is valid
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=["HS256"]
            )
            return True
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
            
    @api_route(version="v1.0.0")
    async def execute_code(self,
                          execution: CodeExecution,
                          background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Execute code in isolated container.
        
        Args:
            execution: Code execution details
            background_tasks: Background tasks runner
            
        Returns:
            Dict[str, Any]: Execution details
        """
        # Generate execution ID
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        # Store execution details
        await self.redis.hset(
            f"execution:{execution_id}",
            mapping={
                "status": "pending",
                "code": execution.code,
                "language": execution.language,
                "task_description": execution.task_description,
                "requirements": ",".join(execution.requirements),
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Add to execution queue
        background_tasks.add_task(
            self._execute_code_task,
            execution_id,
            execution
        )
        
        return {
            "execution_id": execution_id,
            "status": "pending"
        }
        
    async def _execute_code_task(self, execution_id: str, execution: CodeExecution):
        """Execute code in background.
        
        Args:
            execution_id: Execution ID
            execution: Code execution details
        """
        try:
            # Update status
            await self.redis.hset(
                f"execution:{execution_id}",
                "status",
                "running"
            )
            
            # Create container config
            container_config = {
                "image": f"openhands/{execution.language}:latest",
                "command": ["python", "-c", execution.code],
                "environment": {
                    "PYTHONUNBUFFERED": "1"
                },
                "mem_limit": "512m",
                "cpu_quota": 100000,  # 10% of CPU
                "network_disabled": True
            }
            
            # Run container
            container = self.docker.containers.run(
                **container_config,
                detach=True
            )
            
            try:
                # Wait for container
                result = container.wait(timeout=30)
                
                # Get logs
                logs = container.logs().decode()
                
                # Store results
                await self.redis.hset(
                    f"execution:{execution_id}",
                    mapping={
                        "status": "completed" if result["StatusCode"] == 0 else "failed",
                        "exit_code": result["StatusCode"],
                        "output": logs,
                        "completed_at": datetime.now().isoformat()
                    }
                )
                
            finally:
                # Cleanup container
                container.remove(force=True)
                
        except Exception as e:
            # Log error
            self.log_error(e, {
                "execution_id": execution_id,
                "language": execution.language
            })
            
            # Update status
            await self.redis.hset(
                f"execution:{execution_id}",
                mapping={
                    "status": "error",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                }
            )
            
    @api_route(version="v1.0.0")
    async def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get execution details.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Dict[str, Any]: Execution details
            
        Raises:
            HTTPException: If execution not found
        """
        # Get execution details
        execution = await self.redis.hgetall(f"execution:{execution_id}")
        
        if not execution:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found"
            )
            
        return {
            "execution_id": execution_id,
            **execution
        }
        
    @api_route(version="v1.0.0")
    async def build_image(self, build: DockerBuild) -> Dict[str, Any]:
        """Build Docker image.
        
        Args:
            build: Build details
            
        Returns:
            Dict[str, Any]: Build results
        """
        try:
            # Create build context
            context = {
                name: content.encode()
                for name, content in build.context.items()
            }
            
            # Build image
            image, logs = self.docker.images.build(
                fileobj=build.dockerfile.encode(),
                tag=build.tag,
                rm=True,
                forcerm=True,
                buildargs=context
            )
            
            return {
                "image_id": image.id,
                "tag": build.tag,
                "logs": [log.get("stream", "") for log in logs]
            }
            
        except Exception as e:
            self.log_error(e, {"tag": build.tag})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def run_container(self, config: DockerRun) -> Dict[str, Any]:
        """Run Docker container.
        
        Args:
            config: Container configuration
            
        Returns:
            Dict[str, Any]: Container details
        """
        try:
            # Run container
            container = self.docker.containers.run(
                image=config.image,
                command=config.command,
                environment=config.environment,
                ports=config.ports,
                volumes=config.volumes,
                detach=True
            )
            
            return {
                "container_id": container.id,
                "status": container.status
            }
            
        except Exception as e:
            self.log_error(e, {"image": config.image})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def stop_container(self, container_id: str) -> Dict[str, Any]:
        """Stop Docker container.
        
        Args:
            container_id: Container ID
            
        Returns:
            Dict[str, Any]: Operation result
        """
        try:
            container = self.docker.containers.get(container_id)
            container.stop()
            
            return {
                "container_id": container_id,
                "status": "stopped"
            }
            
        except Exception as e:
            self.log_error(e, {"container_id": container_id})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def get_container_logs(self, container_id: str) -> str:
        """Get container logs.
        
        Args:
            container_id: Container ID
            
        Returns:
            str: Container logs
        """
        try:
            container = self.docker.containers.get(container_id)
            return container.logs().decode()
            
        except Exception as e:
            self.log_error(e, {"container_id": container_id})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def deploy_stack(self, deployment: ComposeDeployment) -> Dict[str, Any]:
        """Deploy Docker Compose stack.
        
        Args:
            deployment: Stack deployment details
            
        Returns:
            Dict[str, Any]: Deployment results
        """
        try:
            # Write compose file
            compose_path = f"/tmp/{deployment.stack_name}_compose.yml"
            with open(compose_path, "w") as f:
                f.write(deployment.compose_file)
                
            # Deploy stack
            self.docker.compose.up(
                project_name=deployment.stack_name,
                compose_files=[compose_path],
                environment=deployment.environment,
                detach=True
            )
            
            return {
                "stack_name": deployment.stack_name,
                "status": "deployed"
            }
            
        except Exception as e:
            self.log_error(e, {"stack_name": deployment.stack_name})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def get_stack_status(self, stack_name: str) -> Dict[str, Any]:
        """Get stack status.
        
        Args:
            stack_name: Stack name
            
        Returns:
            Dict[str, Any]: Stack status
        """
        try:
            services = self.docker.compose.ps(
                project_name=stack_name,
                services=True
            )
            
            return {
                "stack_name": stack_name,
                "services": {
                    service.name: {
                        "state": service.state,
                        "status": service.status
                    }
                    for service in services
                }
            }
            
        except Exception as e:
            self.log_error(e, {"stack_name": stack_name})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def scale_service(self,
                          stack_name: str,
                          scale: ServiceScale) -> Dict[str, Any]:
        """Scale stack service.
        
        Args:
            stack_name: Stack name
            scale: Scaling configuration
            
        Returns:
            Dict[str, Any]: Operation result
        """
        try:
            self.docker.compose.scale(
                project_name=stack_name,
                service_scale={scale.service: scale.replicas}
            )
            
            return {
                "stack_name": stack_name,
                "service": scale.service,
                "replicas": scale.replicas
            }
            
        except Exception as e:
            self.log_error(e, {
                "stack_name": stack_name,
                "service": scale.service
            })
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status.
        
        Returns:
            Dict[str, Any]: System status
        """
        try:
            return {
                "docker": {
                    "version": self.docker.version(),
                    "info": self.docker.info()
                },
                "redis": {
                    "connected": bool(self.redis),
                    "info": await self.redis.info() if self.redis else None
                }
            }
            
        except Exception as e:
            self.log_error(e, {})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    @api_route(version="v1.0.0")
    async def get_supported_languages(self) -> List[str]:
        """Get supported programming languages.
        
        Returns:
            List[str]: Supported languages
        """
        try:
            # Get language images
            images = self.docker.images.list(
                filters={"reference": "openhands/*"}
            )
            
            return [
                image.tags[0].split("/")[1].split(":")[0]
                for image in images
                if image.tags
            ]
            
        except Exception as e:
            self.log_error(e, {})
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )