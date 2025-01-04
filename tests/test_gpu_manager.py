import unittest
import torch
import torch.nn as nn
import time
from managers.gpu_manager import (
    GPUMode,
    GPUConfig,
    GPUManager
)

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, size: int = 1000):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

class TestGPUManager(unittest.TestCase):
    """Test GPU manager functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Skip tests if no GPU
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No GPU available")
            
        # Create GPU manager
        cls.config = GPUConfig(
            mode=GPUMode.SINGLE,
            devices=[0],
            memory_fraction=0.5,
            mixed_precision=True
        )
        cls.manager = GPUManager(cls.config)
        
    def setUp(self):
        """Set up each test."""
        # Clear GPU memory
        torch.cuda.empty_cache()
        
    def test_gpu_setup(self):
        """Test GPU setup."""
        self.assertTrue(self.manager.has_gpu)
        self.assertGreater(self.manager.gpu_count, 0)
        self.assertEqual(len(self.manager.gpu_handles), self.manager.gpu_count)
        
    def test_model_preparation(self):
        """Test model preparation."""
        # Create model
        model = SimpleModel()
        prepared_model = self.manager.prepare_model(model)
        
        # Check device
        self.assertTrue(next(prepared_model.parameters()).is_cuda)
        
        # Test forward pass
        x = torch.randn(10, 1000).cuda()
        output = prepared_model(x)
        self.assertTrue(output.is_cuda)
        
    def test_memory_optimization(self):
        """Test memory optimization."""
        # Record initial memory
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate tensors
        tensors = [torch.randn(1000, 1000).cuda() for _ in range(10)]
        
        # Check memory increased
        self.assertGreater(
            torch.cuda.memory_allocated(),
            initial_memory
        )
        
        # Optimize memory
        self.manager.optimize_memory()
        
        # Check memory decreased
        self.assertLess(
            torch.cuda.memory_allocated(),
            torch.cuda.max_memory_allocated()
        )
        
    def test_gpu_monitoring(self):
        """Test GPU monitoring."""
        # Start monitoring
        self.manager.start_monitoring(interval=0.1)
        
        # Create some GPU load
        model = self.manager.prepare_model(SimpleModel())
        x = torch.randn(100, 1000).cuda()
        for _ in range(10):
            _ = model(x)
            
        # Wait for metrics
        time.sleep(0.5)
        
        # Get metrics
        metrics = self.manager.get_gpu_metrics()
        
        # Check metrics
        self.assertIn("utilization", metrics)
        self.assertIn("memory", metrics)
        self.assertIn("temperature", metrics)
        
        # Stop monitoring
        self.manager.stop_monitoring()
        
    def test_memory_profiling(self):
        """Test memory profiling."""
        # Create model and input
        model = SimpleModel()
        input_tensor = torch.randn(10, 1000)
        
        # Profile memory
        profile = self.manager.profile_memory(
            self.manager.prepare_model(model),
            input_tensor.cuda()
        )
        
        # Check profile
        self.assertIn("model_size_mb", profile)
        self.assertIn("activation_mb", profile)
        self.assertIn("peak_mb", profile)
        self.assertIn("total_mb", profile)
        
    def test_inference_optimization(self):
        """Test inference optimization."""
        # Create model
        model = SimpleModel()
        input_tensor = torch.randn(10, 1000)
        
        # Optimize model
        optimized_model = self.manager.optimize_for_inference(
            self.manager.prepare_model(model)
        )
        
        # Test inference
        with torch.no_grad():
            output = optimized_model(input_tensor.cuda())
            
        self.assertTrue(output.is_cuda)
        self.assertEqual(output.shape, (10, 1000))
        
    def test_multi_gpu_setup(self):
        """Test multi-GPU setup."""
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs")
            
        # Create multi-GPU config
        config = GPUConfig(
            mode=GPUMode.DATA_PARALLEL,
            devices=[0, 1],
            memory_fraction=0.5
        )
        manager = GPUManager(config)
        
        # Create model
        model = SimpleModel()
        parallel_model = manager.prepare_model(model)
        
        # Check model type
        self.assertIsInstance(parallel_model, nn.DataParallel)
        
        # Test forward pass
        x = torch.randn(20, 1000).cuda()
        output = parallel_model(x)
        self.assertTrue(output.is_cuda)
        
    def test_memory_stats(self):
        """Test memory statistics."""
        # Get initial stats
        stats = self.manager.get_memory_stats()
        
        # Check stats structure
        self.assertIn("allocated", stats)
        self.assertIn("cached", stats)
        self.assertIn("reserved", stats)
        self.assertIn("active", stats)
        
        # Create some memory usage
        tensors = [torch.randn(1000, 1000).cuda() for _ in range(5)]
        
        # Get updated stats
        new_stats = self.manager.get_memory_stats()
        
        # Check memory increased
        self.assertGreater(
            new_stats["allocated"][0]["bytes"],
            stats["allocated"][0]["bytes"]
        )
        
    def test_error_handling(self):
        """Test error handling."""
        # Test invalid device
        with self.assertRaises(Exception):
            config = GPUConfig(
                mode=GPUMode.SINGLE,
                devices=[100]  # Invalid device
            )
            GPUManager(config)
            
        # Test invalid model
        with self.assertRaises(Exception):
            self.manager.prepare_model(None)
            
    def tearDown(self):
        """Clean up after each test."""
        torch.cuda.empty_cache()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.manager.cleanup()

if __name__ == "__main__":
    unittest.main()