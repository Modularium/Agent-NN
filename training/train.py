"""Training script for agent selector model."""
import os
import argparse
from pathlib import Path
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
from rich.console import Console
from rich.progress import Progress

from training.data_logger import InteractionLogger, create_dataloaders
from training.agent_selector_model import AgentSelectorModel, AgentSelectorTrainer
from utils.logging_util import setup_logger

logger = setup_logger(__name__)
console = Console()

def train(args):
    """Train agent selector model.
    
    Args:
        args: Command line arguments
    """
    # Set up MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "hidden_dims": args.hidden_dims,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout
        })
        
        # Load and prepare data
        data_logger = InteractionLogger(args.log_dir)
        
        with Progress() as progress:
            task = progress.add_task("Loading data...", total=4)
            
            # Prepare training data
            df = data_logger.prepare_training_data(
                min_interactions=args.min_interactions
            )
            progress.update(task, advance=1)
            
            # Create feature and target matrices
            features = data_logger.create_feature_matrix(df)
            targets = data_logger.create_target_matrix(df)
            progress.update(task, advance=1)
            
            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                features,
                targets,
                batch_size=args.batch_size
            )
            progress.update(task, advance=1)
            
            # Initialize model
            input_dim = features.shape[1]
            num_agents = len(data_logger.agent_encoder.classes_)
            num_metrics = targets.shape[1] - 2  # Subtract agent and success columns
            
            model = AgentSelectorModel(
                input_dim=input_dim,
                hidden_dims=args.hidden_dims,
                num_agents=num_agents,
                num_metrics=num_metrics,
                dropout=args.dropout
            )
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate
            )
            
            trainer = AgentSelectorTrainer(model, optimizer)
            progress.update(task, advance=1)
            
        # Training loop
        best_accuracy = 0.0
        
        with Progress() as progress:
            epoch_task = progress.add_task(
                "Training...",
                total=args.num_epochs
            )
            
            for epoch in range(args.num_epochs):
                # Train epoch
                metrics = trainer.train_epoch(train_loader, val_loader)
                
                # Log metrics
                mlflow.log_metrics(
                    metrics,
                    step=epoch
                )
                
                # Save best model
                if metrics["agent_accuracy"] > best_accuracy:
                    best_accuracy = metrics["agent_accuracy"]
                    
                    # Save checkpoint
                    checkpoint_path = os.path.join(
                        args.checkpoint_dir,
                        f"model_epoch_{epoch}.pt"
                    )
                    trainer.save_checkpoint(checkpoint_path)
                    
                    # Log model
                    mlflow.pytorch.log_model(
                        model,
                        "model",
                        registered_model_name=args.model_name
                    )
                    
                # Update progress
                progress.update(epoch_task, advance=1)
                
                # Print metrics
                console.print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
                console.print("Training Metrics:")
                for k, v in metrics.items():
                    if k.endswith("loss"):
                        console.print(f"  {k}: {v:.4f}")
                        
                if "agent_accuracy" in metrics:
                    console.print("\nValidation Metrics:")
                    console.print(
                        f"  Agent Accuracy: {metrics['agent_accuracy']:.4f}"
                    )
                    console.print(
                        f"  Success Accuracy: {metrics['success_accuracy']:.4f}"
                    )
                    console.print(
                        f"  Metrics MSE: {metrics['metrics_mse']:.4f}"
                    )
                    
        # Final evaluation
        final_metrics = trainer.validate_step(
            features,
            targets[:, 0].long(),
            targets[:, 1],
            targets[:, 2:]
        )
        
        mlflow.log_metrics(final_metrics)
        
        console.print("\nTraining Complete!")
        console.print("Final Metrics:")
        for k, v in final_metrics.items():
            console.print(f"  {k}: {v:.4f}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train agent selector model")
    
    # Data arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        default="data/logs",
        help="Directory containing interaction logs"
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=100,
        help="Minimum number of interactions required"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Hidden layer dimensions"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    # MLflow arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="agent_selector",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="agent_selector",
        help="Model name for MLflow registry"
    )
    
    # Output arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Start training
    train(args)

if __name__ == "__main__":
    main()