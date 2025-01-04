from typing import Dict, Any, Optional, List, Tuple
import asyncio
from datetime import datetime, timedelta
import mlflow
from managers.agent_optimizer import AgentOptimizer
from managers.nn_manager import NNManager
from utils.logging_util import LoggerMixin

class EnhancedAgentManager(LoggerMixin):
    """Enhanced manager for agent lifecycle and optimization."""
    
    def __init__(self,
                 optimization_interval: int = 24,  # hours
                 min_performance_threshold: float = 0.7):
        """Initialize agent manager.
        
        Args:
            optimization_interval: Hours between optimizations
            min_performance_threshold: Minimum acceptable performance
        """
        super().__init__()
        self.optimization_interval = timedelta(hours=optimization_interval)
        self.min_performance_threshold = min_performance_threshold
        
        # Initialize managers
        self.optimizer = AgentOptimizer(
            min_performance_threshold=min_performance_threshold
        )
        self.nn_manager = NNManager()
        
        # Track agents
        self.agents: Dict[str, Dict[str, Any]] = {}
        
        # Start optimization loop
        self.optimization_task = asyncio.create_task(
            self._run_optimization_loop()
        )
        
    async def create_new_agent(self,
                             description: str,
                             capabilities: List[str],
                             initial_config: Optional[Dict[str, Any]] = None) -> str:
        """Create new agent with semantic domain mapping.
        
        Args:
            description: Agent description
            capabilities: Agent capabilities
            initial_config: Optional initial configuration
            
        Returns:
            str: Agent identifier
        """
        try:
            # Determine domain
            domain, confidence = await self.optimizer.determine_domain(
                description,
                capabilities
            )
            
            # Generate agent ID
            agent_id = f"{domain}_agent_{len(self.agents) + 1}"
            
            # Get domain knowledge
            domain_docs = await self.optimizer.get_domain_knowledge(domain)
            
            # Get optimized prompts
            prompts = await self.optimizer.optimize_prompts(
                agent_id,
                domain,
                ["general", "specific", "analysis"]
            )
            
            # Create agent configuration
            config = {
                "domain": domain,
                "domain_confidence": confidence,
                "description": description,
                "capabilities": capabilities,
                "prompts": prompts,
                "created_at": datetime.now().isoformat(),
                "last_optimized": None,
                "status": "initializing",
                **(initial_config or {})
            }
            
            # Store agent
            self.agents[agent_id] = config
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"create_{agent_id}"):
                mlflow.log_params({
                    "domain": domain,
                    "domain_confidence": confidence,
                    "num_documents": len(domain_docs)
                })
                
            # Log creation
            self.log_event(
                "agent_created",
                {
                    "agent_id": agent_id,
                    "domain": domain,
                    "confidence": confidence
                }
            )
            
            return agent_id
            
        except Exception as e:
            self.log_error(e, {
                "description": description,
                "operation": "create_agent"
            })
            raise
            
    async def update_agent_metrics(self,
                                 agent_id: str,
                                 metrics: Dict[str, Any]):
        """Update agent performance metrics.
        
        Args:
            agent_id: Agent identifier
            metrics: Performance metrics
        """
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
            
        # Update optimizer metrics
        await self.optimizer.update_agent_metrics(agent_id, metrics)
        
        # Update agent status
        self.agents[agent_id]["last_active"] = datetime.now().isoformat()
        
        # Log update
        self.log_event(
            "metrics_updated",
            {
                "agent_id": agent_id,
                "metrics": metrics
            }
        )
        
    async def optimize_agent(self, agent_id: str) -> Dict[str, Any]:
        """Optimize agent configuration.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
            
        agent = self.agents[agent_id]
        
        try:
            # Evaluate current performance
            evaluation = await self.optimizer.evaluate_agent(agent_id)
            
            if not evaluation["needs_optimization"]:
                return {
                    "status": "skipped",
                    "reason": "performance above threshold",
                    "performance": evaluation["performance"]
                }
                
            # Run optimization
            results = await self.optimizer.optimize_agent(
                agent_id,
                agent["domain"],
                evaluation
            )
            
            # Update agent configuration
            agent["prompts"].update(results["new_prompts"])
            agent["last_optimized"] = datetime.now().isoformat()
            
            # Log optimization
            self.log_event(
                "agent_optimized",
                {
                    "agent_id": agent_id,
                    "optimization_id": results["optimization_id"]
                }
            )
            
            return {
                "status": "optimized",
                "optimization_id": results["optimization_id"],
                "changes": results["changes"]
            }
            
        except Exception as e:
            self.log_error(e, {
                "agent_id": agent_id,
                "operation": "optimize_agent"
            })
            raise
            
    async def _run_optimization_loop(self):
        """Run periodic agent optimization."""
        while True:
            try:
                for agent_id, agent in self.agents.items():
                    # Check if optimization is needed
                    if agent["last_optimized"]:
                        last_opt = datetime.fromisoformat(
                            agent["last_optimized"]
                        )
                        if datetime.now() - last_opt < self.optimization_interval:
                            continue
                            
                    # Run optimization
                    await self.optimize_agent(agent_id)
                    
            except Exception as e:
                self.log_error(e, {"operation": "optimization_loop"})
                
            # Wait for next interval
            await asyncio.sleep(3600)  # Check every hour
            
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get agent information.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dict[str, Any]: Agent information
        """
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
            
        return self.agents[agent_id].copy()
        
    def get_domain_agents(self, domain: str) -> List[str]:
        """Get agents for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List[str]: Agent identifiers
        """
        return [
            agent_id
            for agent_id, agent in self.agents.items()
            if agent["domain"] == domain
        ]
        
    def get_capable_agents(self, capability: str) -> List[str]:
        """Get agents with capability.
        
        Args:
            capability: Required capability
            
        Returns:
            List[str]: Agent identifiers
        """
        return [
            agent_id
            for agent_id, agent in self.agents.items()
            if capability in agent["capabilities"]
        ]
        
    async def cleanup(self):
        """Clean up resources."""
        # Cancel optimization loop
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass