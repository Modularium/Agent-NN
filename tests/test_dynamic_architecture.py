import unittest
from unittest.mock import patch, MagicMock
import torch
import tempfile
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from nn_models.dynamic_architecture import (
    LayerType,
    LayerConfig,
    DynamicLayer,
    DynamicArchitecture,
    ArchitectureOptimizer
)

class TestDynamicArchitecture(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('nn_models.dynamic_architecture.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment
        
        # Create test data
        self.batch_size = 4
        self.input_dim = 10
        self.output_dim = 2
        
        self.inputs = torch.randn(self.batch_size, self.input_dim)
        self.targets = torch.randint(0, self.output_dim, (self.batch_size,))
        
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        import shutil
        shutil.rmtree(self.test_dir)
        
    def test_dynamic_layer(self):
        """Test dynamic layer creation and forward pass."""
        # Test linear layer
        config = LayerConfig(
            layer_type=LayerType.LINEAR,
            input_dim=10,
            output_dim=5
        )
        layer = DynamicLayer(config)
        x = torch.randn(2, 10)
        output = layer(x)
        self.assertEqual(output.shape, (2, 5))
        
        # Test attention layer
        config = LayerConfig(
            layer_type=LayerType.ATTENTION,
            input_dim=12,  # Must be divisible by num_heads
            output_dim=12,
            params={"num_heads": 3}
        )
        layer = DynamicLayer(config)
        x = torch.randn(2, 3, 12)  # (batch, seq_len, dim)
        output = layer(x)
        self.assertEqual(output.shape, (2, 3, 12))
        
    def test_dynamic_architecture(self):
        """Test dynamic architecture creation and forward pass."""
        # Create architecture with dimensions divisible by 3
        input_dim = 12  # Must be divisible by num_heads
        output_dim = 12
        model = DynamicArchitecture(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        # Create test inputs
        inputs = torch.randn(self.batch_size, input_dim)
        
        # Test forward pass
        output = model(inputs)
        self.assertEqual(
            output.shape,
            (self.batch_size, output_dim)
        )
        
        # Test architecture adaptation
        model.adapt_architecture(
            task_requirements={
                "complexity": "high",
                "attention_needed": True
            },
            performance_metrics={
                "accuracy": 0.5
            }
        )
        
        # Check if layers were added
        self.assertGreater(len(model.layers), 3)
        
    def test_architecture_save_load(self):
        """Test architecture saving and loading."""
        # Create and modify architecture with dimensions divisible by 3
        input_dim = 12  # Must be divisible by num_heads
        output_dim = 12
        model = DynamicArchitecture(
            input_dim=input_dim,
            output_dim=output_dim
        )
        model.adapt_architecture(
            task_requirements={"complexity": "high"},
            performance_metrics={"accuracy": 0.5}
        )
        
        # Save architecture
        save_path = os.path.join(self.test_dir, "architecture.json")
        model.save_architecture(save_path)
        
        # Load architecture
        loaded_model = DynamicArchitecture.load_architecture(save_path)
        
        # Compare architectures
        self.assertEqual(
            len(model.layers),
            len(loaded_model.layers)
        )
        
        # Create test inputs
        inputs = torch.randn(self.batch_size, input_dim)
        
        # Test forward pass
        output1 = model(inputs)
        output2 = loaded_model(inputs)
        self.assertEqual(output1.shape, output2.shape)
        
    def test_architecture_optimizer(self):
        """Test architecture optimization."""
        # Create model and optimizer with dimensions divisible by 3
        input_dim = 12  # Must be divisible by num_heads
        output_dim = 12
        model = DynamicArchitecture(
            input_dim=input_dim,
            output_dim=output_dim
        )
        optimizer = ArchitectureOptimizer(model)
        
        # Create test data
        inputs = torch.randn(self.batch_size, input_dim)
        targets = torch.randn(self.batch_size, output_dim)
        
        # Create data loaders
        dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)
        
        # Test training step
        batch = (inputs, targets)
        metrics = optimizer.train_step(batch)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        
        # Test evaluation
        metrics = optimizer.evaluate(val_loader)
        self.assertIn("val_loss", metrics)
        self.assertIn("val_accuracy", metrics)
        
        # Test full optimization
        results = optimizer.optimize_architecture(
            train_loader,
            val_loader,
            task_requirements={
                "complexity": "medium",
                "accuracy_threshold": 0.9
            },
            num_epochs=2
        )
        
        self.assertIn("best_metrics", results)
        self.assertIn("final_architecture", results)
        self.assertIsInstance(results["final_architecture"], list)
        self.assertGreater(len(results["final_architecture"]), 0)
        self.assertIn("type", results["final_architecture"][0])
        
    def test_layer_types(self):
        """Test different layer types."""
        input_dim = 10
        output_dim = 5
        batch_size = 2
        seq_len = 3
        
        layer_configs = [
            (LayerType.LINEAR, (batch_size, input_dim)),
            (LayerType.CONV1D, (batch_size, input_dim, seq_len)),
            (LayerType.LSTM, (batch_size, seq_len, input_dim)),
            (LayerType.GRU, (batch_size, seq_len, input_dim))
        ]
        
        for config_data in layer_configs:
            if len(config_data) == 3:
                layer_type, input_shape, dim = config_data
                config = LayerConfig(
                    layer_type=layer_type,
                    input_dim=dim,
                    output_dim=dim,
                    params={"num_heads": 3}
                )
            else:
                layer_type, input_shape = config_data
                config = LayerConfig(
                    layer_type=layer_type,
                    input_dim=input_dim,
                    output_dim=output_dim
                )
            layer = DynamicLayer(config)
            
            # Test forward pass
            x = torch.randn(*input_shape)
            output = layer(x)
            self.assertIsNotNone(output)
            
    def test_architecture_adaptation(self):
        """Test architecture adaptation strategies."""
        model = DynamicArchitecture(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        # Test different adaptation scenarios
        scenarios = [
            {
                "requirements": {
                    "complexity": "high",
                    "attention_needed": True
                },
                "metrics": {"accuracy": 0.5},
                "expected_changes": 1
            },
            {
                "requirements": {
                    "complexity": "medium",
                    "sequence_data": True
                },
                "metrics": {"accuracy": 0.7},
                "expected_changes": 1
            },
            {
                "requirements": {
                    "complexity": "low",
                    "attention_needed": False
                },
                "metrics": {"accuracy": 0.95},
                "expected_changes": 0
            }
        ]
        
        for scenario in scenarios:
            initial_layers = len(model.layers)
            model.adapt_architecture(
                scenario["requirements"],
                scenario["metrics"]
            )
            changes = len(model.layers) - initial_layers
            self.assertEqual(changes, scenario["expected_changes"])
            
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid layer type
        with self.assertRaises(ValueError):
            config = LayerConfig(
                layer_type="invalid",
                input_dim=10,
                output_dim=5
            )
            DynamicLayer(config)
            
        # Test invalid architecture save path
        model = DynamicArchitecture(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        with self.assertRaises(Exception):
            model.save_architecture("/invalid/path/arch.json")
            
        # Test loading invalid architecture
        with self.assertRaises(Exception):
            DynamicArchitecture.load_architecture("/invalid/path/arch.json")

if __name__ == '__main__':
    unittest.main()