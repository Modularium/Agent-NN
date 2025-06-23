"""API server for Agent-NN."""
import os
import asyncio
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime

from .endpoints import APIEndpoints
from .models import SystemMetrics
from managers.monitoring_system import MonitoringSystem
from utils.logging_util import LoggerMixin

class APIServer(LoggerMixin):
    """FastAPI server for Agent-NN."""
    
    def __init__(self,
                host: str = "0.0.0.0",
                port: int = 8000,
                log_level: str = "info"):
        """Initialize API server.
        
        Args:
            host: Server host
            port: Server port
            log_level: Logging level
        """
        super().__init__()
        self.host = host
        self.port = port
        self.log_level = log_level
        
        # Initialize FastAPI
        self.app = FastAPI(
            title="Agent-NN API",
            description="API for multi-agent neural network system",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Initialize endpoints
        self.endpoints = APIEndpoints()
        
        # Add routes
        self.app.include_router(
            self.endpoints.router,
            prefix="/api/v2"
        )
        
        # Add error handlers
        self._add_error_handlers()
        
    def _add_error_handlers(self):
        """Add custom error handlers."""
        @self.app.exception_handler(Exception)
        async def generic_error_handler(request, exc):
            """Handle generic exceptions."""
            self.log_error(exc, {
                "path": request.url.path,
                "method": request.method
            })
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    async def start(self):
        """Start API server."""
        # Start monitoring
        self.endpoints.monitoring.start()
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def stop(self):
        """Stop API server."""
        # Stop monitoring
        self.endpoints.monitoring.stop()
        
        # Stop server
        # Note: uvicorn doesn't provide a clean way to stop programmatically
        pass
        
    async def get_metrics(self) -> SystemMetrics:
        """Get current system metrics.
        
        Returns:
            SystemMetrics: Current metrics
        """
        return await self.endpoints.monitoring.get_metrics()
        
    async def healthcheck(self) -> bool:
        """Check server health.
        
        Returns:
            bool: Whether server is healthy
        """
        try:
            metrics = await self.get_metrics()
            return (
                metrics.cpu_usage < 90 and
                metrics.memory_usage < 90 and
                metrics.task_queue_size < 1000
            )
        except:
            return False

def create_app() -> FastAPI:
    """Create FastAPI application.
    
    Returns:
        FastAPI: Application instance
    """
    server = APIServer()
    return server.app

if __name__ == "__main__":
    server = APIServer()
    asyncio.run(server.start())