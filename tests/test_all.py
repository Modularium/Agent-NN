import pytest
import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import asyncio
import threading
import tempfile
import shutil
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
from tests.test_gpu_manager import TestGPUManager
from tests.test_model_registry import TestModelRegistry
from tests.test_online_learning import TestOnlineLearning
from tests.test_ab_testing import TestABTesting
from tests.integration.test_system_integration import TestSystemIntegration
from tests.test_dynamic_architecture import TestDynamicArchitecture

# Test fixtures
@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()

@pytest.fixture(scope="session")
def sample_model():
    """Create sample model for testing."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
            
    return SimpleModel()

@pytest.fixture(scope="session")
def sample_data():
    """Create sample data for testing."""
    return {
        "input": torch.randn(100, 10),
        "target": torch.randn(100, 1)
    }

# Test suites
def test_gpu_manager(gpu_available, sample_model):
    """Test GPU manager functionality."""
    if not gpu_available:
        pytest.skip("No GPU available")
        
    # Run all GPU manager tests
    test_suite = TestGPUManager()
    test_suite.setUp()
    
    try:
        # Test model preparation
        test_suite.test_model_preparation()
        
        # Test memory optimization
        test_suite.test_memory_optimization()
        
        # Test GPU monitoring
        test_suite.test_gpu_monitoring()
        
        # Test memory profiling
        test_suite.test_memory_profiling()
        
        # Test inference optimization
        test_suite.test_inference_optimization()
        
    finally:
        test_suite.tearDown()

def test_model_registry(temp_dir, sample_model):
    """Test model registry functionality."""
    # Run all model registry tests
    test_suite = TestModelRegistry()
    test_suite.setUp()
    
    try:
        # Test model version
        test_suite.test_model_version()
        
        # Test model registration
        test_suite.test_register_model()
        
        # Test version limit
        test_suite.test_version_limit()
        
        # Test model loading
        test_suite.test_load_model()
        
        # Test best version selection
        test_suite.test_best_version()
        
    finally:
        test_suite.tearDown()

def test_online_learning(sample_model, sample_data):
    """Test online learning functionality."""
    # Run all online learning tests
    test_suite = TestOnlineLearning()
    test_suite.setUp()
    
    try:
        # Test streaming buffer
        test_suite.test_streaming_buffer()
        
        # Test streaming dataset
        test_suite.test_streaming_dataset()
        
        # Test adaptive learning rate
        test_suite.test_adaptive_learning_rate()
        
        # Test online learner
        test_suite.test_online_learner()
        
        # Test model improvement
        test_suite.test_model_improvement()
        
    finally:
        test_suite.tearDown()

def test_ab_testing(sample_model):
    """Test A/B testing functionality."""
    # Run all A/B testing tests
    test_suite = TestABTesting()
    test_suite.setUp()
    test_suite.model_a = sample_model
    test_suite.model_b = sample_model
    
    try:
        # Test test creation
        test_suite.test_test_creation()
        
        # Test test completion
        test_suite.test_test_completion()
        
        # Test result analysis
        test_suite.test_result_analysis()
        
        # Test state persistence
        test_suite.test_state_persistence()
        
        # Test error handling
        test_suite.test_error_handling()
        
    finally:
        test_suite.tearDown()
        
    # End MLflow run
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

@pytest.mark.asyncio
async def test_system_integration(temp_dir, sample_model, sample_data):
    """Test system integration."""
    # End any active MLflow run
    try:
        import mlflow
        while mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass
        
    # Convert sample_data to numpy arrays
    import numpy as np
    sample_data = {
        key: value.detach().numpy().tolist() if hasattr(value, 'detach') else value
        for key, value in sample_data.items()
    }
        
    # Run all integration tests
    test_suite = TestSystemIntegration()
    test_suite.setUpClass()
    test_suite.temp_dir = temp_dir
    test_suite.model = sample_model
    test_suite.data = sample_data
    
    try:
        # Test system configuration
        await test_suite.test_system_configuration()
        
        # End MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
        
        # Test backup and restore
        await test_suite.test_backup_restore()
        
        # End MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
        
        # Test cache operations
        await test_suite.test_cache_operations()
        
        # End MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
        
        # Test model management
        await test_suite.test_model_management()
        
        # End MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
        
        # Test knowledge base
        await test_suite.test_knowledge_base()
        
        # End MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
        
        # Test system integration
        await test_suite.test_system_integration()
        
        # End MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass
        
        # All tests passed
        assert True
        
    finally:
        test_suite.tearDownClass()
        
        # End MLflow run
        try:
            while mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass

def test_dynamic_architecture(gpu_available, sample_model):
    """Test dynamic architecture functionality."""
    if not gpu_available:
        pytest.skip("No GPU available")
        
    # Run all dynamic architecture tests
    test_suite = TestDynamicArchitecture()
    test_suite.setUp()
    
    try:
        # Test dynamic layer
        test_suite.test_dynamic_layer()
        
        # Test dynamic architecture
        test_suite.test_dynamic_architecture()
        
        # Test architecture save/load
        test_suite.test_architecture_save_load()
        
        # Test architecture optimizer
        test_suite.test_architecture_optimizer()
        
        # Test layer types
        test_suite.test_layer_types()
        
    finally:
        test_suite.tearDown()

# Main test runner
if __name__ == "__main__":
    # Configure pytest
    pytest_args = [
        "--verbose",
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--asyncio-mode=auto",
        "-v"
    ]
    
    # Run tests
    pytest.main(pytest_args)