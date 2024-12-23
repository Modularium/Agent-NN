"""MLflow experiment tracking for model training and evaluation."""
from typing import Dict, Any, Optional, List
import mlflow
from datetime import datetime
from config import MLFLOW_TRACKING_URI

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        """Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.experiment_name = experiment_name
        
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
            
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            mlflow.ActiveRun: The active run context
        """
        return mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
    def log_agent_selection(self, 
                          task_description: str,
                          chosen_agent: str,
                          available_agents: List[str],
                          agent_scores: Dict[str, float],
                          execution_result: Dict[str, Any]) -> None:
        """Log an agent selection event.
        
        Args:
            task_description: Description of the task
            chosen_agent: Name of the chosen agent
            available_agents: List of available agents
            agent_scores: Scores for each agent
            execution_result: Result of task execution
        """
        with self.start_run("agent_selection") as run:
            # Log task and agent information
            mlflow.log_param("task_description", task_description)
            mlflow.log_param("chosen_agent", chosen_agent)
            mlflow.log_param("available_agents", available_agents)
            
            # Log agent scores
            for agent, score in agent_scores.items():
                mlflow.log_metric(f"score_{agent}", score)
            
            # Log execution metrics
            mlflow.log_metric("execution_time", execution_result.get("execution_time", 0))
            mlflow.log_metric("success", 1 if execution_result.get("success") else 0)
            
            if "error" in execution_result:
                mlflow.log_param("error", execution_result["error"])
                
    def log_model_update(self,
                        model_type: str,
                        parameters: Dict[str, Any],
                        metrics: Dict[str, float],
                        artifacts: Optional[Dict[str, str]] = None) -> None:
        """Log a model update event.
        
        Args:
            model_type: Type of model being updated
            parameters: Model parameters
            metrics: Model performance metrics
            artifacts: Optional paths to model artifacts
        """
        with self.start_run(f"{model_type}_update") as run:
            # Log parameters
            for name, value in parameters.items():
                mlflow.log_param(name, value)
                
            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
                    
    def log_evaluation_results(self,
                             evaluation_type: str,
                             results: Dict[str, Any]) -> None:
        """Log evaluation results.
        
        Args:
            evaluation_type: Type of evaluation performed
            results: Evaluation results
        """
        with self.start_run(f"{evaluation_type}_evaluation") as run:
            # Log all metrics from results
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, value)
                    
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Optional[Dict[str, Any]]:
        """Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether to minimize (True) or maximize (False) the metric
            
        Returns:
            Dict containing the best run information, or None if no runs found
        """
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"metrics.{metric_name} IS NOT NULL",
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1
        )
        
        if len(runs) == 0:
            return None
            
        best_run = runs.iloc[0]
        return {
            "run_id": best_run.run_id,
            "metrics": {col.split(".")[-1]: best_run[col] 
                       for col in runs.columns 
                       if col.startswith("metrics.")},
            "params": {col.split(".")[-1]: best_run[col] 
                      for col in runs.columns 
                      if col.startswith("params.")}
        }