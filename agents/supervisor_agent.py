from typing import Dict, Any, Optional, List
from datetime import datetime
import mlflow
from managers.agent_manager import AgentManager
from managers.nn_manager import NNManager

class SupervisorAgent:
    def __init__(self):
        """Initialize the supervisor agent with its managers."""
        self.agent_manager = AgentManager()
        self.nn_manager = NNManager()
        
        # Keep track of task execution history
        self.task_history = []

    def execute_task(self, task_description: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Execute a task by selecting and delegating to the most appropriate agent.
        
        Args:
            task_description: Description of the task to execute
            context: Optional additional context for the task
            
        Returns:
            Dict containing execution results and metadata
        """
        start_time = datetime.now()
        
        with mlflow.start_run(run_name="task_execution") as run:
            # Log task information
            mlflow.log_param("task_description", task_description)
            if context:
                mlflow.log_param("context", context)
            
            # Get available agents
            available_agents = self.agent_manager.get_all_agents()
            mlflow.log_param("available_agents", available_agents)
            
            # Select best agent using the neural network
            chosen_agent_name = self.nn_manager.predict_best_agent(
                task_description,
                available_agents
            )
            
            # Create new agent if no suitable agent found
            if chosen_agent_name is None:
                chosen_agent = self.agent_manager.create_new_agent(task_description)
                chosen_agent_name = f"{chosen_agent.name}_agent_{len(available_agents)+1}"
                mlflow.log_param("new_agent_created", True)
            else:
                chosen_agent = self.agent_manager.get_agent(chosen_agent_name)
                mlflow.log_param("new_agent_created", False)
            
            mlflow.log_param("chosen_agent", chosen_agent_name)
            
            try:
                # Execute the task
                result = chosen_agent.execute_task(task_description, context)
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Prepare execution record
                execution_record = {
                    "task_description": task_description,
                    "context": context,
                    "chosen_agent": chosen_agent_name,
                    "result": result,
                    "execution_time": execution_time,
                    "timestamp": start_time.isoformat(),
                    "success": True,
                    "error": None
                }
                
                # Log metrics
                mlflow.log_metric("execution_time", execution_time)
                mlflow.log_metric("success", 1)
                
                # Update model based on success
                self._update_model(task_description, chosen_agent_name, success_score=1.0)
                
            except Exception as e:
                # Handle execution failure
                execution_time = (datetime.now() - start_time).total_seconds()
                execution_record = {
                    "task_description": task_description,
                    "context": context,
                    "chosen_agent": chosen_agent_name,
                    "result": None,
                    "execution_time": execution_time,
                    "timestamp": start_time.isoformat(),
                    "success": False,
                    "error": str(e)
                }
                
                # Log failure metrics
                mlflow.log_metric("execution_time", execution_time)
                mlflow.log_metric("success", 0)
                mlflow.log_param("error", str(e))
                
                # Update model based on failure
                self._update_model(task_description, chosen_agent_name, success_score=0.0)
            
            # Store execution record
            self.task_history.append(execution_record)
            
            return execution_record
    
    def _update_model(self, task_description: str, chosen_agent: str, success_score: float):
        """Update the neural network model based on task execution results.
        
        Args:
            task_description: Description of the executed task
            chosen_agent: Name of the agent that executed the task
            success_score: Score indicating how well the task was executed (0-1)
        """
        self.nn_manager.update_model(task_description, chosen_agent, success_score)
    
    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get status information about a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dict containing agent status information
        """
        # Get agent metadata
        metadata = self.agent_manager.get_agent_metadata(agent_name)
        
        # Calculate agent performance metrics from history
        agent_tasks = [task for task in self.task_history 
                      if task["chosen_agent"] == agent_name]
        
        if agent_tasks:
            success_rate = sum(1 for task in agent_tasks if task["success"]) / len(agent_tasks)
            avg_execution_time = sum(task["execution_time"] for task in agent_tasks) / len(agent_tasks)
        else:
            success_rate = 0
            avg_execution_time = 0
        
        return {
            **metadata,
            "total_tasks": len(agent_tasks),
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "last_task_timestamp": agent_tasks[-1]["timestamp"] if agent_tasks else None
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task execution history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of execution records, most recent first
        """
        return sorted(self.task_history, 
                     key=lambda x: x["timestamp"], 
                     reverse=True)[:limit]
