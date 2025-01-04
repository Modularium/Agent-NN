import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
import time
from datetime import datetime, timedelta
from managers.monitoring_system import (
    MetricType,
    AlertSeverity,
    MetricConfig,
    Alert,
    MetricBuffer,
    MonitoringSystem
)

class TestMonitoringSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('managers.monitoring_system.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock experiment
        self.mock_experiment = MagicMock()
        self.mock_experiment.experiment_id = "test_experiment"
        self.mock_mlflow.set_experiment.return_value = self.mock_experiment
        
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize monitoring system
        self.monitor = MonitoringSystem(check_interval=0.1)
        
        # Create test metric
        self.test_metric = MetricConfig(
            name="test_metric",
            type=MetricType.CUSTOM,
            unit="count",
            thresholds={
                AlertSeverity.WARNING: 80.0,
                AlertSeverity.CRITICAL: 90.0
            }
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        import shutil
        shutil.rmtree(self.test_dir)
        
        # Stop monitoring
        if self.monitor.running:
            self.monitor.stop()
            
    def test_metric_buffer(self):
        """Test metric buffer."""
        buffer = MetricBuffer(self.test_metric)
        
        # Add values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            buffer.add(value)
            
        # Check statistics
        stats = buffer.get_statistics()
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        
        # Check windowed statistics
        window_stats = buffer.get_statistics(
            window=timedelta(seconds=0.1)
        )
        self.assertEqual(window_stats["count"], 5)
        
    def test_alert_generation(self):
        """Test alert generation."""
        buffer = MetricBuffer(self.test_metric)
        
        # Add normal value
        buffer.add(50.0)
        self.assertIsNone(buffer.check_thresholds())
        
        # Add warning value
        buffer.add(85.0)
        alert = buffer.check_thresholds()
        self.assertEqual(alert.severity, AlertSeverity.WARNING)
        
        # Add critical value
        buffer.add(95.0)
        alert = buffer.check_thresholds()
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        
    def test_monitoring_system(self):
        """Test monitoring system."""
        # Add metric
        self.monitor.add_metric(self.test_metric)
        
        # Add alert handler
        alerts = []
        def alert_handler(alert):
            alerts.append(alert)
            
        self.monitor.add_alert_handler(alert_handler)
        
        # Start monitoring
        self.monitor.start()
        
        # Record values
        self.monitor.record_metric("test_metric", 50.0)
        self.monitor.record_metric("test_metric", 85.0)
        self.monitor.record_metric("test_metric", 95.0)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check alerts
        self.assertEqual(len(alerts), 2)  # Warning and Critical
        self.assertEqual(alerts[0].severity, AlertSeverity.WARNING)
        self.assertEqual(alerts[1].severity, AlertSeverity.CRITICAL)
        
        # Get statistics
        stats = self.monitor.get_metric_statistics("test_metric")
        self.assertGreater(stats["mean"], 0)
        
    def test_system_metrics(self):
        """Test system metrics collection."""
        # Start monitoring
        self.monitor.start()
        
        # Wait for collection
        time.sleep(0.5)
        
        # Check metrics
        stats = self.monitor.get_all_statistics()
        self.assertIn("cpu_usage", stats)
        self.assertIn("memory_usage", stats)
        
        # Check values
        self.assertGreater(stats["cpu_usage"]["mean"], 0)
        self.assertGreater(stats["memory_usage"]["mean"], 0)
        
    def test_state_persistence(self):
        """Test state saving and loading."""
        # Add metric and values
        self.monitor.add_metric(self.test_metric)
        self.monitor.record_metric("test_metric", 50.0)
        self.monitor.record_metric("test_metric", 60.0)
        
        # Save state
        state_path = os.path.join(self.test_dir, "state.json")
        self.monitor.save_state(state_path)
        
        # Create new monitor
        new_monitor = MonitoringSystem()
        new_monitor.load_state(state_path)
        
        # Check loaded state
        self.assertIn("test_metric", new_monitor.metrics)
        stats = new_monitor.get_metric_statistics("test_metric")
        self.assertEqual(stats["mean"], 55.0)
        
    def test_metric_windowing(self):
        """Test metric windowing."""
        self.monitor.add_metric(self.test_metric)
        
        # Add values at different times
        self.monitor.record_metric("test_metric", 50.0)
        time.sleep(0.2)
        self.monitor.record_metric("test_metric", 60.0)
        
        # Get recent window
        stats = self.monitor.get_metric_statistics(
            "test_metric",
            window=timedelta(seconds=0.1)
        )
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["mean"], 60.0)
        
    def test_error_handling(self):
        """Test error handling."""
        # Test unknown metric
        with self.assertRaises(ValueError):
            self.monitor.record_metric("unknown", 50.0)
            
        # Test duplicate metric
        with self.assertRaises(ValueError):
            self.monitor.add_metric(self.test_metric)
            self.monitor.add_metric(self.test_metric)
            
        # Test invalid statistics
        with self.assertRaises(ValueError):
            self.monitor.get_metric_statistics("unknown")
            
    def test_alert_handlers(self):
        """Test alert handler management."""
        # Add multiple handlers
        alerts1 = []
        alerts2 = []
        
        def handler1(alert):
            alerts1.append(alert)
            
        def handler2(alert):
            alerts2.append(alert)
            
        self.monitor.add_alert_handler(handler1)
        self.monitor.add_alert_handler(handler2)
        
        # Add metric and start
        self.monitor.add_metric(self.test_metric)
        self.monitor.start()
        
        # Generate alert
        self.monitor.record_metric("test_metric", 95.0)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check both handlers received alert
        self.assertEqual(len(alerts1), 1)
        self.assertEqual(len(alerts2), 1)
        self.assertEqual(
            alerts1[0].severity,
            alerts2[0].severity
        )
        
    def test_system_shutdown(self):
        """Test system shutdown."""
        # Start monitoring
        self.monitor.start()
        self.assertTrue(self.monitor.running)
        
        # Stop monitoring
        self.monitor.stop()
        self.assertFalse(self.monitor.running)
        
        # Check threads stopped
        for thread in self.monitor.threads:
            self.assertFalse(thread.is_alive())

if __name__ == '__main__':
    unittest.main()