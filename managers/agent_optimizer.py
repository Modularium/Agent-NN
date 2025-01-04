from typing import Dict, Any, Optional, List, Union, Tuple
import asyncio
import torch
from datetime import datetime, timedelta
import mlflow
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from utils.logging_util import LoggerMixin
from config.llm_config import OPENAI_CONFIG

class AgentOptimizer(LoggerMixin):
    """Manager for agent optimization and improvement."""
    
    def __init__(self,
                 knowledge_base_path: str = "knowledge_bases",
                 min_performance_threshold: float = 0.7,
                 evaluation_interval: int = 24,  # hours
                 embedding_batch_size: int = 100):
        """Initialize agent optimizer.
        
        Args:
            knowledge_base_path: Path to knowledge bases
            min_performance_threshold: Minimum acceptable performance
            evaluation_interval: Hours between evaluations
            embedding_batch_size: Batch size for embeddings
        """
        super().__init__()
        self.knowledge_base_path = knowledge_base_path
        self.min_performance_threshold = min_performance_threshold
        self.evaluation_interval = timedelta(hours=evaluation_interval)
        self.embedding_batch_size = embedding_batch_size
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_CONFIG["api_key"]
        )
        
        # Initialize vector store
        self.domain_store = Chroma(
            collection_name="domain_knowledge",
            embedding_function=self.embeddings,
            persist_directory=knowledge_base_path
        )
        
        # Track agent performance
        self.agent_metrics: Dict[str, List[Dict[str, Any]]] = {}
        
        # Track optimization history
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # MLflow experiment
        self.experiment = mlflow.set_experiment("agent_optimization")
        
    async def determine_domain(self,
                             description: str,
                             capabilities: List[str]) -> Tuple[str, float]:
        """Determine agent domain using embeddings.
        
        Args:
            description: Agent description
            capabilities: Agent capabilities
            
        Returns:
            Tuple[str, float]: Domain and confidence score
        """
        # Create query embedding
        query = f"{description}\nCapabilities: {', '.join(capabilities)}"
        
        # Search domain knowledge
        results = self.domain_store.similarity_search_with_score(
            query,
            k=3
        )
        
        if not results:
            return "general", 0.0
            
        # Get most similar domain
        domains = {}
        for doc, score in results:
            domain = doc.metadata.get("domain", "general")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(score)
            
        # Calculate average score per domain
        domain_scores = {
            domain: sum(scores) / len(scores)
            for domain, scores in domains.items()
        }
        
        # Get best domain
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain
        
    async def get_domain_knowledge(self,
                                 domain: str,
                                 limit: int = 10) -> List[Document]:
        """Get domain-specific knowledge.
        
        Args:
            domain: Domain name
            limit: Maximum documents
            
        Returns:
            List[Document]: Domain documents
        """
        # Search domain knowledge
        return self.domain_store.similarity_search(
            f"domain: {domain}",
            k=limit,
            filter={"domain": domain}
        )
        
    async def optimize_prompts(self,
                             agent_id: str,
                             domain: str,
                             task_types: List[str]) -> Dict[str, str]:
        """Optimize agent prompts.
        
        Args:
            agent_id: Agent identifier
            domain: Agent domain
            task_types: Types of tasks
            
        Returns:
            Dict[str, str]: Optimized prompts
        """
        # Get domain knowledge
        domain_docs = await self.get_domain_knowledge(domain)
        
        # Create prompt templates
        prompts = {}
        for task_type in task_types:
            # Get relevant examples
            examples = [
                doc for doc in domain_docs
                if task_type in doc.metadata.get("task_types", [])
            ]
            
            if not examples:
                continue
                
            # Create template
            template = f"""You are a specialized agent for {domain} tasks.
            
            Task Type: {task_type}
            
            Domain Knowledge:
            {{domain_knowledge}}
            
            Task Description:
            {{task_description}}
            
            Please provide a detailed response following these guidelines:
            1. Use domain-specific terminology
            2. Reference relevant knowledge
            3. Explain your reasoning
            4. Provide actionable insights
            
            Response:"""
            
            prompts[task_type] = template
            
        return prompts
        
    async def update_agent_metrics(self,
                                 agent_id: str,
                                 metrics: Dict[str, Any]):
        """Update agent performance metrics.
        
        Args:
            agent_id: Agent identifier
            metrics: Performance metrics
        """
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = []
            
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Add to metrics
        self.agent_metrics[agent_id].append(metrics)
        
        # Log to MLflow
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=f"agent_{agent_id}"
        ):
            mlflow.log_metrics(metrics)
            
    async def evaluate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Evaluate agent performance.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if agent_id not in self.agent_metrics:
            return {
                "status": "unknown",
                "performance": 0.0,
                "needs_optimization": True
            }
            
        # Get recent metrics
        recent_metrics = [
            m for m in self.agent_metrics[agent_id]
            if datetime.fromisoformat(m["timestamp"]) > 
            datetime.now() - self.evaluation_interval
        ]
        
        if not recent_metrics:
            return {
                "status": "inactive",
                "performance": 0.0,
                "needs_optimization": True
            }
            
        # Calculate performance metrics
        avg_quality = sum(
            m.get("response_quality", 0)
            for m in recent_metrics
        ) / len(recent_metrics)
        
        success_rate = sum(
            1 for m in recent_metrics
            if m.get("task_success", False)
        ) / len(recent_metrics)
        
        user_satisfaction = sum(
            m.get("user_satisfaction", 0)
            for m in recent_metrics
        ) / len(recent_metrics)
        
        # Calculate overall performance
        performance = (
            0.4 * avg_quality +
            0.4 * success_rate +
            0.2 * user_satisfaction
        )
        
        return {
            "status": "active",
            "performance": performance,
            "metrics": {
                "response_quality": avg_quality,
                "success_rate": success_rate,
                "user_satisfaction": user_satisfaction
            },
            "needs_optimization": performance < self.min_performance_threshold
        }
        
    async def optimize_agent(self,
                           agent_id: str,
                           domain: str,
                           current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize underperforming agent.
        
        Args:
            agent_id: Agent identifier
            domain: Agent domain
            current_performance: Current performance metrics
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Start MLflow run
            with mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=f"optimize_{agent_id}"
            ) as run:
                # Log current performance
                mlflow.log_metrics(current_performance["metrics"])
                
                # Get additional domain knowledge
                new_docs = await self.get_domain_knowledge(
                    domain,
                    limit=20
                )
                
                # Optimize prompts
                new_prompts = await self.optimize_prompts(
                    agent_id,
                    domain,
                    ["general", "specific", "analysis"]
                )
                
                # Record optimization
                optimization = {
                    "optimization_id": optimization_id,
                    "timestamp": datetime.now().isoformat(),
                    "type": "full",
                    "changes": {
                        "new_documents": len(new_docs),
                        "new_prompts": len(new_prompts)
                    },
                    "previous_performance": current_performance
                }
                
                if agent_id not in self.optimization_history:
                    self.optimization_history[agent_id] = []
                self.optimization_history[agent_id].append(optimization)
                
                # Log optimization
                self.log_event(
                    "agent_optimized",
                    {
                        "agent_id": agent_id,
                        "optimization_id": optimization_id,
                        "changes": optimization["changes"]
                    }
                )
                
                return {
                    "optimization_id": optimization_id,
                    "new_documents": new_docs,
                    "new_prompts": new_prompts,
                    "run_id": run.info.run_id
                }
                
        except Exception as e:
            self.log_error(e, {
                "agent_id": agent_id,
                "operation": "optimize_agent"
            })
            raise
            
    async def get_optimization_history(self,
                                     agent_id: str,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """Get agent optimization history.
        
        Args:
            agent_id: Agent identifier
            limit: Maximum records
            
        Returns:
            List[Dict[str, Any]]: Optimization history
        """
        if agent_id not in self.optimization_history:
            return []
            
        history = sorted(
            self.optimization_history[agent_id],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        return history[:limit]
        
    def get_agent_performance_trend(self,
                                  agent_id: str,
                                  days: int = 7) -> Dict[str, List[float]]:
        """Get agent performance trend.
        
        Args:
            agent_id: Agent identifier
            days: Number of days
            
        Returns:
            Dict[str, List[float]]: Performance metrics over time
        """
        if agent_id not in self.agent_metrics:
            return {}
            
        cutoff = datetime.now() - timedelta(days=days)
        metrics = [
            m for m in self.agent_metrics[agent_id]
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not metrics:
            return {}
            
        # Group by day
        daily_metrics = {}
        for metric in metrics:
            date = datetime.fromisoformat(
                metric["timestamp"]
            ).date().isoformat()
            
            if date not in daily_metrics:
                daily_metrics[date] = []
            daily_metrics[date].append(metric)
            
        # Calculate daily averages
        trend = {
            "response_quality": [],
            "success_rate": [],
            "user_satisfaction": []
        }
        
        for date in sorted(daily_metrics.keys()):
            day_metrics = daily_metrics[date]
            
            trend["response_quality"].append(
                sum(m.get("response_quality", 0) for m in day_metrics) /
                len(day_metrics)
            )
            
            trend["success_rate"].append(
                sum(1 for m in day_metrics if m.get("task_success", False)) /
                len(day_metrics)
            )
            
            trend["user_satisfaction"].append(
                sum(m.get("user_satisfaction", 0) for m in day_metrics) /
                len(day_metrics)
            )
            
        return trend