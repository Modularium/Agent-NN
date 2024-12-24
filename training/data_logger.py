"""Data logging and preprocessing for model training."""
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from utils.logging_util import setup_logger

logger = setup_logger(__name__)

class InteractionLogger:
    """Logger for agent interactions and performance."""
    
    def __init__(self, log_dir: str = "data/logs"):
        """Initialize interaction logger.
        
        Args:
            log_dir: Directory for storing logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.interactions_dir = self.log_dir / "interactions"
        self.metrics_dir = self.log_dir / "metrics"
        self.features_dir = self.log_dir / "features"
        
        for directory in [self.interactions_dir, self.metrics_dir, self.features_dir]:
            directory.mkdir(exist_ok=True)
            
        # Initialize label encoders
        self.agent_encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()
        self.task_encoder = LabelEncoder()
        
        # Load existing encoders if available
        self._load_encoders()
        
        # Initialize tokenizer for text embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def log_interaction(self,
                       task_description: str,
                       chosen_agent: str,
                       success: bool,
                       metrics: Dict[str, float],
                       metadata: Optional[Dict[str, Any]] = None):
        """Log an agent interaction.
        
        Args:
            task_description: Description of the task
            chosen_agent: Name of chosen agent
            success: Whether the interaction was successful
            metrics: Performance metrics
            metadata: Optional additional metadata
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "chosen_agent": chosen_agent,
            "success": success,
            "metrics": metrics,
            **(metadata or {})
        }
        
        # Save interaction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.interactions_dir / f"interaction_{timestamp}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(interaction, f, indent=2)
                
            logger.info(f"Logged interaction to {file_path}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
            
    def log_metrics(self,
                   agent: str,
                   metrics: Dict[str, float],
                   metadata: Optional[Dict[str, Any]] = None):
        """Log agent performance metrics.
        
        Args:
            agent: Agent name
            metrics: Performance metrics
            metadata: Optional additional metadata
        """
        metric_data = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "metrics": metrics,
            **(metadata or {})
        }
        
        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.metrics_dir / f"metrics_{timestamp}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(metric_data, f, indent=2)
                
            logger.info(f"Logged metrics to {file_path}")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            
    def prepare_training_data(self,
                            min_interactions: int = 100,
                            success_only: bool = True) -> pd.DataFrame:
        """Prepare training data from logged interactions.
        
        Args:
            min_interactions: Minimum number of interactions required
            success_only: Whether to use only successful interactions
            
        Returns:
            DataFrame containing training data
        """
        interactions = []
        
        # Load all interaction logs
        for file_path in self.interactions_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    interaction = json.load(f)
                    
                if success_only and not interaction["success"]:
                    continue
                    
                interactions.append(interaction)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                
        if len(interactions) < min_interactions:
            raise ValueError(
                f"Not enough interactions: {len(interactions)} < {min_interactions}"
            )
            
        # Convert to DataFrame
        df = pd.DataFrame(interactions)
        
        # Extract features
        df["task_embedding"] = df["task_description"].apply(self._get_text_embedding)
        df["agent_encoded"] = self.agent_encoder.fit_transform(df["chosen_agent"])
        
        if "domain" in df.columns:
            df["domain_encoded"] = self.domain_encoder.fit_transform(df["domain"])
            
        if "task_type" in df.columns:
            df["task_encoded"] = self.task_encoder.fit_transform(df["task_type"])
            
        # Save encoders
        self._save_encoders()
        
        return df
        
    def create_feature_matrix(self, df: pd.DataFrame) -> torch.Tensor:
        """Create feature matrix from DataFrame.
        
        Args:
            df: DataFrame with interaction data
            
        Returns:
            Tensor containing feature matrix
        """
        features = []
        
        # Task embeddings
        task_embeddings = np.stack(df["task_embedding"].values)
        features.append(task_embeddings)
        
        # Agent encodings (one-hot)
        agent_onehot = pd.get_dummies(df["agent_encoded"]).values
        features.append(agent_onehot)
        
        # Domain encodings if available
        if "domain_encoded" in df.columns:
            domain_onehot = pd.get_dummies(df["domain_encoded"]).values
            features.append(domain_onehot)
            
        # Task type encodings if available
        if "task_encoded" in df.columns:
            task_onehot = pd.get_dummies(df["task_encoded"]).values
            features.append(task_onehot)
            
        # Concatenate all features
        feature_matrix = np.concatenate(features, axis=1)
        
        return torch.FloatTensor(feature_matrix)
        
    def create_target_matrix(self, df: pd.DataFrame) -> torch.Tensor:
        """Create target matrix from DataFrame.
        
        Args:
            df: DataFrame with interaction data
            
        Returns:
            Tensor containing target matrix
        """
        # Extract success and metrics
        success = df["success"].astype(float).values
        metrics = pd.DataFrame(df["metrics"].tolist()).values
        
        # Combine targets
        targets = np.concatenate([
            success.reshape(-1, 1),
            metrics
        ], axis=1)
        
        return torch.FloatTensor(targets)
        
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using BERT tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            numpy array containing embedding
        """
        # Tokenize and get BERT embedding
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Use mean pooling of token embeddings
        embeddings = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        
        # Apply attention mask and get mean
        masked_embeddings = embeddings * attention_mask
        mean_embedding = masked_embeddings.mean(axis=1)
        
        return mean_embedding[0]
        
    def _save_encoders(self):
        """Save label encoders."""
        encoders = {
            "agent": self.agent_encoder,
            "domain": self.domain_encoder,
            "task": self.task_encoder
        }
        
        for name, encoder in encoders.items():
            try:
                file_path = self.features_dir / f"{name}_encoder.json"
                with open(file_path, 'w') as f:
                    json.dump({
                        "classes": encoder.classes_.tolist()
                    }, f)
            except Exception as e:
                logger.error(f"Error saving {name} encoder: {str(e)}")
                
    def _load_encoders(self):
        """Load label encoders."""
        encoders = {
            "agent": self.agent_encoder,
            "domain": self.domain_encoder,
            "task": self.task_encoder
        }
        
        for name, encoder in encoders.items():
            try:
                file_path = self.features_dir / f"{name}_encoder.json"
                if file_path.exists():
                    with open(file_path) as f:
                        data = json.load(f)
                        encoder.classes_ = np.array(data["classes"])
            except Exception as e:
                logger.error(f"Error loading {name} encoder: {str(e)}")

class AgentInteractionDataset(Dataset):
    """Dataset for agent interaction data."""
    
    def __init__(self,
                 features: torch.Tensor,
                 targets: torch.Tensor):
        """Initialize dataset.
        
        Args:
            features: Feature matrix
            targets: Target matrix
        """
        self.features = features
        self.targets = targets
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.features)
        
    def __getitem__(self, idx: int) -> tuple:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (features, targets)
        """
        return self.features[idx], self.targets[idx]
        
def create_dataloaders(features: torch.Tensor,
                      targets: torch.Tensor,
                      batch_size: int = 32,
                      train_ratio: float = 0.8) -> tuple:
    """Create train and validation dataloaders.
    
    Args:
        features: Feature matrix
        targets: Target matrix
        batch_size: Batch size
        train_ratio: Ratio of training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create dataset
    dataset = AgentInteractionDataset(features, targets)
    
    # Split indices
    indices = torch.randperm(len(dataset))
    train_size = int(train_ratio * len(dataset))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices)
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    return train_loader, val_loader