"""Agent improvement and fine-tuning system."""
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import mlflow
from rich.console import Console
from rich.progress import Progress

from utils.logging_util import setup_logger
from llm_models.specialized_llm import SpecializedLLM
from llm_models.llm_backend import LLMBackendManager, LLMBackendType
from agents.worker_agent import WorkerAgent
from agents.agent_communication import AgentCommunicationHub
from agents.domain_knowledge import DomainKnowledgeManager
from training.data_logger import InteractionLogger

logger = setup_logger(__name__)
console = Console()

class AgentImprover:
    """System for improving and fine-tuning agents."""
    
    def __init__(self,
                 communication_hub: AgentCommunicationHub,
                 knowledge_manager: DomainKnowledgeManager,
                 improvement_interval: int = 3600,  # 1 hour
                 min_interactions: int = 50):
        """Initialize agent improver.
        
        Args:
            communication_hub: Communication hub
            knowledge_manager: Knowledge manager
            improvement_interval: Seconds between improvements
            min_interactions: Minimum interactions before improvement
        """
        self.communication_hub = communication_hub
        self.knowledge_manager = knowledge_manager
        self.improvement_interval = improvement_interval
        self.min_interactions = min_interactions
        
        # Initialize components
        self.llm = SpecializedLLM("agent_improver")
        self.backend_manager = LLMBackendManager()
        self.interaction_logger = InteractionLogger()
        
        # Track improvement history
        self.improvement_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Start MLflow
        mlflow.set_experiment("agent_improvement")
        
    async def start_improvement_loop(self):
        """Start continuous improvement loop."""
        while True:
            try:
                # Get active agents
                agents = self._get_active_agents()
                
                for agent in agents:
                    # Check if improvement needed
                    if await self._should_improve_agent(agent.name):
                        await self.improve_agent(agent)
                        
            except Exception as e:
                logger.error(f"Error in improvement loop: {str(e)}")
                
            await asyncio.sleep(self.improvement_interval)
            
    async def improve_agent(self, agent: WorkerAgent) -> bool:
        """Improve an agent.
        
        Args:
            agent: Agent to improve
            
        Returns:
            bool: True if improvement successful
        """
        try:
            with mlflow.start_run(run_name=f"improve_{agent.name}"):
                # Get performance data
                performance = await self._analyze_performance(agent.name)
                mlflow.log_metrics(performance)
                
                # Identify improvement areas
                improvements = await self._identify_improvements(
                    agent.name,
                    performance
                )
                mlflow.log_dict(improvements, "improvements.json")
                
                # Apply improvements
                success = await self._apply_improvements(agent, improvements)
                
                if success:
                    # Log improvement
                    self._log_improvement(agent.name, improvements)
                    mlflow.log_metric("improvement_success", 1.0)
                    return True
                    
                mlflow.log_metric("improvement_success", 0.0)
                return False
                
        except Exception as e:
            logger.error(f"Error improving agent {agent.name}: {str(e)}")
            return False
            
    async def fine_tune_agent(self,
                            agent: WorkerAgent,
                            training_data: List[Dict[str, Any]]) -> bool:
        """Fine-tune an agent's LLM.
        
        Args:
            agent: Agent to fine-tune
            training_data: Training examples
            
        Returns:
            bool: True if fine-tuning successful
        """
        try:
            with mlflow.start_run(run_name=f"finetune_{agent.name}"):
                # Prepare training data
                formatted_data = self._format_training_data(training_data)
                mlflow.log_dict(formatted_data, "training_data.json")
                
                # Fine-tune LLM
                success = await self._fine_tune_llm(
                    agent.name,
                    formatted_data
                )
                
                if success:
                    mlflow.log_metric("finetune_success", 1.0)
                    return True
                    
                mlflow.log_metric("finetune_success", 0.0)
                return False
                
        except Exception as e:
            logger.error(f"Error fine-tuning agent {agent.name}: {str(e)}")
            return False
            
    async def _analyze_performance(self,
                                 agent_name: str) -> Dict[str, float]:
        """Analyze agent performance.
        
        Args:
            agent_name: Agent name
            
        Returns:
            Dict containing performance metrics
        """
        # Get recent interactions
        interactions = self._get_agent_interactions(agent_name)
        
        if not interactions:
            return {}
            
        # Calculate metrics
        success_rate = np.mean([int(i["success"]) for i in interactions])
        
        # Calculate precision, recall, f1
        y_true = [int(i["success"]) for i in interactions]
        y_pred = [
            1 if i["metrics"].get("confidence", 0) > 0.5 else 0
            for i in interactions
        ]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='binary'
        )
        
        metrics = {
            "success_rate": success_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_response_time": np.mean([
                i["metrics"].get("response_time", 0)
                for i in interactions
            ]),
            "avg_confidence": np.mean([
                i["metrics"].get("confidence", 0)
                for i in interactions
            ])
        }
        
        return metrics
        
    async def _identify_improvements(self,
                                   agent_name: str,
                                   performance: Dict[str, float]) -> Dict[str, Any]:
        """Identify needed improvements.
        
        Args:
            agent_name: Agent name
            performance: Performance metrics
            
        Returns:
            Dict containing improvement suggestions
        """
        # Get recent failures
        failures = self._get_agent_failures(agent_name)
        
        prompt = f"""Analyze agent performance and suggest improvements:

Performance Metrics:
{json.dumps(performance, indent=2)}

Recent Failures:
{json.dumps(failures, indent=2)}

Suggest improvements in JSON format with fields:
1. knowledge_updates
2. prompt_updates
3. llm_updates
4. tool_updates
5. priority (high/medium/low)"""

        try:
            response = self.llm.generate_response(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error identifying improvements: {str(e)}")
            return {}
            
    async def _apply_improvements(self,
                                agent: WorkerAgent,
                                improvements: Dict[str, Any]) -> bool:
        """Apply improvements to agent.
        
        Args:
            agent: Agent to improve
            improvements: Improvement suggestions
            
        Returns:
            bool: True if improvements applied successfully
        """
        try:
            # Update knowledge if needed
            if "knowledge_updates" in improvements:
                knowledge = improvements["knowledge_updates"]
                agent.ingest_knowledge(knowledge)
                
            # Update prompts if needed
            if "prompt_updates" in improvements:
                prompts = improvements["prompt_updates"]
                agent.llm.update_prompts(prompts)
                
            # Update LLM if needed
            if "llm_updates" in improvements:
                updates = improvements["llm_updates"]
                await self._update_llm(agent, updates)
                
            # Update tools if needed
            if "tool_updates" in improvements:
                tools = improvements["tool_updates"]
                await self._update_tools(agent, tools)
                
            return True
            
        except Exception as e:
            logger.error(f"Error applying improvements: {str(e)}")
            return False
            
    async def _fine_tune_llm(self,
                            agent_name: str,
                            training_data: List[Dict[str, Any]]) -> bool:
        """Fine-tune LLM with training data.
        
        Args:
            agent_name: Agent name
            training_data: Training examples
            
        Returns:
            bool: True if fine-tuning successful
        """
        try:
            # This is a placeholder for actual fine-tuning
            # In practice, you'd use the appropriate API for your LLM
            logger.info(f"Fine-tuning not implemented for {agent_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error fine-tuning LLM: {str(e)}")
            return False
            
    def _get_agent_interactions(self,
                              agent_name: str,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent agent interactions.
        
        Args:
            agent_name: Agent name
            limit: Maximum number of interactions
            
        Returns:
            List of interactions
        """
        interactions = []
        
        for file in self.interaction_logger.interactions_dir.glob("*.json"):
            try:
                with open(file) as f:
                    interaction = json.load(f)
                    if interaction["chosen_agent"] == agent_name:
                        interactions.append(interaction)
            except Exception as e:
                logger.error(f"Error loading interaction: {str(e)}")
                
        # Sort by timestamp and limit
        interactions.sort(
            key=lambda x: x["timestamp"],
            reverse=True
        )
        return interactions[:limit]
        
    def _get_agent_failures(self,
                          agent_name: str,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent agent failures.
        
        Args:
            agent_name: Agent name
            limit: Maximum number of failures
            
        Returns:
            List of failed interactions
        """
        interactions = self._get_agent_interactions(agent_name)
        failures = [i for i in interactions if not i["success"]]
        return failures[:limit]
        
    def _format_training_data(self,
                            data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format data for fine-tuning.
        
        Args:
            data: Raw training data
            
        Returns:
            List of formatted examples
        """
        formatted = []
        
        for example in data:
            formatted.append({
                "prompt": example.get("task_description", ""),
                "completion": example.get("response", ""),
                "metadata": {
                    "success": example.get("success", False),
                    "metrics": example.get("metrics", {})
                }
            })
            
        return formatted
        
    def _log_improvement(self,
                        agent_name: str,
                        improvements: Dict[str, Any]):
        """Log improvement attempt.
        
        Args:
            agent_name: Agent name
            improvements: Applied improvements
        """
        if agent_name not in self.improvement_history:
            self.improvement_history[agent_name] = []
            
        self.improvement_history[agent_name].append({
            "timestamp": datetime.now().isoformat(),
            "improvements": improvements
        })
        
    def _get_active_agents(self) -> List[WorkerAgent]:
        """Get list of active agents.
        
        Returns:
            List of active agents
        """
        agents = []
        
        for msg in self.communication_hub.message_history:
            if msg.sender not in agents:
                agents.append(msg.sender)
                
        return agents
        
    async def _should_improve_agent(self, agent_name: str) -> bool:
        """Check if agent should be improved.
        
        Args:
            agent_name: Agent name
            
        Returns:
            bool: True if improvement needed
        """
        # Check number of interactions
        interactions = self._get_agent_interactions(agent_name)
        if len(interactions) < self.min_interactions:
            return False
            
        # Check recent performance
        performance = await self._analyze_performance(agent_name)
        if not performance:
            return False
            
        # Check if performance is below thresholds
        return (
            performance.get("success_rate", 1.0) < 0.8 or
            performance.get("f1_score", 1.0) < 0.7
        )
        
    def show_improvement_history(self, agent_name: Optional[str] = None):
        """Show improvement history.
        
        Args:
            agent_name: Optional agent name to filter
        """
        table = Table(title="Improvement History")
        table.add_column("Agent")
        table.add_column("Timestamp")
        table.add_column("Improvements")
        table.add_column("Priority")
        
        agents = (
            [agent_name] if agent_name
            else self.improvement_history.keys()
        )
        
        for agent in agents:
            if agent not in self.improvement_history:
                continue
                
            for improvement in self.improvement_history[agent]:
                table.add_row(
                    agent,
                    improvement["timestamp"],
                    ", ".join(improvement["improvements"].keys()),
                    improvement["improvements"].get("priority", "unknown")
                )
                
        console.print(table)