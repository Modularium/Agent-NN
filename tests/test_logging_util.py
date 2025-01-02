import unittest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
import shutil
from datetime import datetime
from dataclasses import dataclass
from utils.logging_util import LoggerMixin, CustomJSONEncoder

@dataclass
class TestData:
    """Test dataclass for JSON encoding."""
    name: str
    value: float
    timestamp: datetime

class TestComponent(LoggerMixin):
    """Test component that uses LoggerMixin."""
    def __init__(self):
        super().__init__()

class TestLoggingUtil(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.old_log_dir = os.environ.get('LOG_DIR')
        os.environ['LOG_DIR'] = self.test_dir
        
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        self.mock_mlflow.active_run.return_value = None
        
        # Initialize test component
        self.component = TestComponent()

    def tearDown(self):
        """Clean up after tests."""
        # Restore original log directory
        if self.old_log_dir:
            os.environ['LOG_DIR'] = self.old_log_dir
        else:
            del os.environ['LOG_DIR']
            
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        
        # Stop MLflow mock
        self.mlflow_patcher.stop()

    def test_json_encoder(self):
        """Test custom JSON encoder."""
        test_data = TestData(
            name="test",
            value=1.23,
            timestamp=datetime(2024, 1, 1, 12, 0)
        )
        
        # Encode and decode
        encoded = json.dumps(test_data, cls=CustomJSONEncoder)
        decoded = json.loads(encoded)
        
        # Check values
        self.assertEqual(decoded["name"], "test")
        self.assertEqual(decoded["value"], 1.23)
        self.assertEqual(decoded["timestamp"], "2024-01-01T12:00:00")

    def test_log_event(self):
        """Test event logging."""
        # Log test event
        event_data = {
            "action": "test",
            "value": 42,
            "nested": {"key": "value"}
        }
        metrics = {"metric1": 1.0, "metric2": 2.0}
        
        self.component.log_event("test_event", event_data, metrics)
        
        # Check MLflow calls
        self.mock_mlflow.log_metrics.assert_called_once_with(metrics)
        self.mock_mlflow.set_tags.assert_called_with({
            'event.action': 'test',
            'event.value': '42',
            'event.nested.key': 'value'
        })

    def test_log_error(self):
        """Test error logging."""
        # Create test error
        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = {"task": "test_task", "status": "failed"}
            self.component.log_error(e, context)
        
        # Check MLflow calls
        self.mock_mlflow.set_tag.assert_any_call("error_type", "ValueError")
        self.mock_mlflow.set_tag.assert_any_call("error_message", "Test error")
        self.mock_mlflow.set_tags.assert_called_with({
            'error_context.task': 'test_task',
            'error_context.status': 'failed'
        })

    def test_log_model_performance(self):
        """Test model performance logging."""
        metrics = {
            "accuracy": 0.95,
            "loss": 0.05
        }
        metadata = {
            "model_type": "test_model",
            "parameters": {"param1": 1, "param2": 2}
        }
        
        self.component.log_model_performance("test_model", metrics, metadata)
        
        # Check MLflow calls
        self.mock_mlflow.log_metrics.assert_called_with(metrics)
        self.mock_mlflow.set_tags.assert_called()  # Check tags were set

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested_dict = {
            "level1": {
                "level2a": {
                    "level3": "value"
                },
                "level2b": 42
            },
            "top": "level"
        }
        
        flattened = self.component._flatten_dict(nested_dict, prefix="test")
        
        # Check flattened structure
        self.assertEqual(flattened["test.level1.level2a.level3"], "value")
        self.assertEqual(flattened["test.level1.level2b"], "42")
        self.assertEqual(flattened["test.top"], "level")

    def test_mlflow_lifecycle(self):
        """Test MLflow run lifecycle."""
        # Check run was started in __init__
        self.mock_mlflow.start_run.assert_called_once()
        
        # End run
        self.component.end_run()
        self.mock_mlflow.end_run.assert_called_once()

if __name__ == '__main__':
    unittest.main()