import unittest
import asyncio
import time
import os
import tempfile
import numpy as np
import torch
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Any
from managers.system_manager import SystemManager
from managers.cache_manager import CacheManager
from managers.model_manager import ModelManager
from managers.knowledge_manager import KnowledgeManager
from utils.logging_util import LoggerMixin

class PerformanceMetrics:
    """Performance metrics collector."""
    
    def __init__(self):
        """Initialize metrics."""
        self.latencies = []
        self.throughputs = []
        self.error_rates = []
        self.memory_usages = []
        self.cpu_usages = []
        self.gpu_usages = []
        self.gpu_memory_usages = []
        self.start_time = None
        self.end_time = None
        
        # Initialize GPU monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self.gpu_count)
            ]
            self.has_gpu = True
        except:
            self.has_gpu = False
            self.gpu_count = 0
            self.gpu_handles = []
        
    def start(self):
        """Start metrics collection."""
        self.start_time = time.time()
        
    def stop(self):
        """Stop metrics collection."""
        self.end_time = time.time()
        
    def add_latency(self, latency: float):
        """Add latency measurement."""
        self.latencies.append(latency)
        
    def add_throughput(self, throughput: float):
        """Add throughput measurement."""
        self.throughputs.append(throughput)
        
    def add_error_rate(self, error_rate: float):
        """Add error rate measurement."""
        self.error_rates.append(error_rate)
        
    def add_memory_usage(self, memory_mb: float):
        """Add memory usage measurement."""
        self.memory_usages.append(memory_mb)
        
    def add_cpu_usage(self, cpu_percent: float):
        """Add CPU usage measurement."""
        self.cpu_usages.append(cpu_percent)
        
    def add_gpu_usage(self, gpu_percent: float):
        """Add GPU usage measurement."""
        self.gpu_usages.append(gpu_percent)
        
    def add_gpu_memory_usage(self, memory_mb: float):
        """Add GPU memory usage measurement."""
        self.gpu_memory_usages.append(memory_mb)
        
    def get_gpu_metrics(self) -> Dict[str, List[Dict[str, float]]]:
        """Get GPU metrics.
        
        Returns:
            Dict[str, List[Dict[str, float]]]: GPU metrics per device
        """
        if not self.has_gpu:
            return {}
            
        try:
            import pynvml
            metrics = {
                "utilization": [],
                "memory": [],
                "temperature": [],
                "power": []
            }
            
            for handle in self.gpu_handles:
                # Get GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["utilization"].append({
                    "gpu": util.gpu,
                    "memory": util.memory
                })
                
                # Get memory info
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics["memory"].append({
                    "total": mem.total / (1024 * 1024),  # MB
                    "used": mem.used / (1024 * 1024),
                    "free": mem.free / (1024 * 1024)
                })
                
                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["temperature"].append({
                    "gpu": temp
                })
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                    metrics["power"].append({
                        "usage": power
                    })
                except:
                    metrics["power"].append({
                        "usage": 0
                    })
                    
            return metrics
            
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            return {}
        
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.
        
        Returns:
            Dict[str, Any]: Metrics summary
        """
        duration = self.end_time - self.start_time
        
        return {
            "duration_seconds": duration,
            "latency": {
                "mean": np.mean(self.latencies),
                "p50": np.percentile(self.latencies, 50),
                "p95": np.percentile(self.latencies, 95),
                "p99": np.percentile(self.latencies, 99)
            },
            "throughput": {
                "mean": np.mean(self.throughputs),
                "peak": max(self.throughputs)
            },
            "error_rate": {
                "mean": np.mean(self.error_rates),
                "max": max(self.error_rates)
            },
            "memory_mb": {
                "mean": np.mean(self.memory_usages),
                "peak": max(self.memory_usages)
            },
            "cpu_percent": {
                "mean": np.mean(self.cpu_usages),
                "peak": max(self.cpu_usages)
            },
            "gpu": self.get_gpu_metrics() if self.has_gpu else {}
        }

class TestSystemPerformance(unittest.TestCase, LoggerMixin):
    """System performance tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directories
        cls.test_dir = tempfile.mkdtemp()
        cls.data_dir = os.path.join(cls.test_dir, "data")
        cls.model_dir = os.path.join(cls.test_dir, "models")
        cls.backup_dir = os.path.join(cls.test_dir, "backups")
        cls.config_dir = os.path.join(cls.test_dir, "config")
        
        # Create directories
        for d in [cls.data_dir, cls.model_dir, cls.backup_dir, cls.config_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Initialize managers
        cls.system_manager = SystemManager(
            data_dir=cls.data_dir,
            backup_dir=cls.backup_dir,
            config_file=os.path.join(cls.config_dir, "system.json")
        )
        
        cls.cache_manager = CacheManager(
            max_size=1024,  # 1GB
            cleanup_interval=1
        )
        
        cls.model_manager = ModelManager(
            model_dir=cls.model_dir,
            cache_dir=os.path.join(cls.test_dir, "cache")
        )
        
        cls.knowledge_manager = KnowledgeManager(
            data_dir=cls.data_dir
        )
        
        # Initialize metrics
        cls.metrics = PerformanceMetrics()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Stop managers
        cls.cache_manager.stop()
        
        # Remove test directory
        import shutil
        shutil.rmtree(cls.test_dir)
        
    async def test_concurrent_model_loading(self):
        """Test concurrent model loading performance."""
        # Create test models
        num_models = 10
        models = []
        for i in range(num_models):
            model = torch.nn.Linear(10, 1)
            path = os.path.join(self.model_dir, f"model_{i}.pt")
            torch.save(model.state_dict(), path)
            models.append({
                "name": f"model_{i}",
                "path": path
            })
            
        # Test concurrent loading
        self.metrics.start()
        
        async def load_model(model_info):
            start = time.time()
            try:
                await self.model_manager.load_model(
                    name=model_info["name"],
                    type="nn",
                    source="local",
                    config={"path": model_info["path"]}
                )
                latency = time.time() - start
                self.metrics.add_latency(latency)
                return True
            except Exception as e:
                self.log_error(e)
                return False
                
        # Load models concurrently
        tasks = [load_model(model) for model in models]
        results = await asyncio.gather(*tasks)
        
        # Calculate metrics
        success_rate = sum(results) / len(results)
        self.metrics.add_error_rate(1 - success_rate)
        
        # Get resource usage
        system_metrics = self.system_manager.get_system_metrics()
        self.metrics.add_memory_usage(system_metrics["memory_percent"])
        self.metrics.add_cpu_usage(system_metrics["cpu_percent"])
        
        self.metrics.stop()
        
        # Log results
        self.log_event(
            "model_loading_test",
            self.metrics.get_summary()
        )
        
        # Verify performance
        summary = self.metrics.get_summary()
        self.assertLess(summary["latency"]["p95"], 5.0)  # 95% under 5s
        self.assertGreater(success_rate, 0.95)  # 95% success
        
    async def test_cache_performance(self):
        """Test cache performance under load."""
        # Create test data
        num_entries = 1000
        data_size = 1000  # 1000 elements per tensor
        
        self.metrics.start()
        
        # Test cache operations
        async def cache_operations():
            start = time.time()
            try:
                # Create random tensor
                tensor = torch.randn(data_size)
                key = f"tensor_{int(time.time() * 1000)}"
                
                # Cache operations
                self.cache_manager.set(key, tensor)
                cached = self.cache_manager.get(key)
                self.cache_manager.delete(key)
                
                latency = time.time() - start
                self.metrics.add_latency(latency)
                return cached is not None
                
            except Exception as e:
                self.log_error(e)
                return False
                
        # Run concurrent operations
        tasks = [cache_operations() for _ in range(num_entries)]
        results = await asyncio.gather(*tasks)
        
        # Calculate metrics
        success_rate = sum(results) / len(results)
        self.metrics.add_error_rate(1 - success_rate)
        
        # Get cache stats
        stats = self.cache_manager.get_stats()
        self.metrics.add_memory_usage(stats["total_size"] / (1024 * 1024))
        
        self.metrics.stop()
        
        # Log results
        self.log_event(
            "cache_performance_test",
            self.metrics.get_summary()
        )
        
        # Verify performance
        summary = self.metrics.get_summary()
        self.assertLess(summary["latency"]["p99"], 0.1)  # 99% under 100ms
        self.assertGreater(success_rate, 0.99)  # 99% success
        
    async def test_knowledge_base_search(self):
        """Test knowledge base search performance."""
        # Create test knowledge base
        kb_name = "performance_kb"
        await self.knowledge_manager.create_knowledge_base(
            name=kb_name,
            domain="test",
            sources=[]
        )
        
        # Add test documents
        num_docs = 100
        doc_size = 1000  # characters
        
        async def add_document(i: int):
            content = f"Test document {i} " * (doc_size // len(f"Test document {i} "))
            return await self.knowledge_manager.process_document(
                kb_name,
                f"doc_{i}.txt",
                content.encode()
            )
            
        # Add documents
        doc_tasks = [add_document(i) for i in range(num_docs)]
        await asyncio.gather(*doc_tasks)
        
        # Test search performance
        self.metrics.start()
        
        async def search_kb():
            start = time.time()
            try:
                results = self.knowledge_manager.search_knowledge_base(
                    kb_name,
                    "test document",
                    limit=10
                )
                latency = time.time() - start
                self.metrics.add_latency(latency)
                return len(results) > 0
                
            except Exception as e:
                self.log_error(e)
                return False
                
        # Run concurrent searches
        num_searches = 100
        search_tasks = [search_kb() for _ in range(num_searches)]
        results = await asyncio.gather(*search_tasks)
        
        # Calculate metrics
        success_rate = sum(results) / len(results)
        self.metrics.add_error_rate(1 - success_rate)
        
        # Get resource usage
        system_metrics = self.system_manager.get_system_metrics()
        self.metrics.add_memory_usage(system_metrics["memory_percent"])
        self.metrics.add_cpu_usage(system_metrics["cpu_percent"])
        
        self.metrics.stop()
        
        # Log results
        self.log_event(
            "kb_search_test",
            self.metrics.get_summary()
        )
        
        # Verify performance
        summary = self.metrics.get_summary()
        self.assertLess(summary["latency"]["p95"], 1.0)  # 95% under 1s
        self.assertGreater(success_rate, 0.95)  # 95% success
        
    async def test_system_backup_restore(self):
        """Test backup and restore performance."""
        # Create test data
        num_files = 100
        file_size = 1024  # 1KB per file
        
        # Create test files
        for i in range(num_files):
            path = os.path.join(self.data_dir, f"file_{i}.txt")
            with open(path, "wb") as f:
                f.write(os.urandom(file_size))
                
        # Test backup performance
        self.metrics.start()
        
        # Create backup
        start = time.time()
        backup_info = await self.system_manager.create_backup(
            include_models=True,
            include_data=True
        )
        backup_latency = time.time() - start
        self.metrics.add_latency(backup_latency)
        
        # Delete test files
        for i in range(num_files):
            path = os.path.join(self.data_dir, f"file_{i}.txt")
            os.remove(path)
            
        # Test restore performance
        start = time.time()
        await self.system_manager.restore_backup(backup_info["backup_id"])
        restore_latency = time.time() - start
        self.metrics.add_latency(restore_latency)
        
        # Verify restoration
        success = all(
            os.path.exists(os.path.join(self.data_dir, f"file_{i}.txt"))
            for i in range(num_files)
        )
        
        self.metrics.add_error_rate(0 if success else 1)
        
        # Get resource usage
        system_metrics = self.system_manager.get_system_metrics()
        self.metrics.add_memory_usage(system_metrics["memory_percent"])
        self.metrics.add_cpu_usage(system_metrics["cpu_percent"])
        
        self.metrics.stop()
        
        # Log results
        self.log_event(
            "backup_restore_test",
            self.metrics.get_summary()
        )
        
        # Verify performance
        summary = self.metrics.get_summary()
        self.assertLess(backup_latency, 30.0)  # Backup under 30s
        self.assertLess(restore_latency, 30.0)  # Restore under 30s
        self.assertTrue(success)  # Successful restoration
        
    def test_system_stress(self):
        """Test system under stress."""
        # Configure stress test
        duration = 60  # 1 minute
        num_threads = 4
        
        self.metrics.start()
        
        def stress_worker():
            try:
                # Create and cache random tensors
                while time.time() - self.metrics.start_time < duration:
                    start = time.time()
                    
                    # Random operation
                    op = np.random.choice([
                        "cache",
                        "model",
                        "knowledge",
                        "system"
                    ])
                    
                    if op == "cache":
                        # Cache operation
                        tensor = torch.randn(1000)
                        key = f"stress_{time.time()}"
                        self.cache_manager.set(key, tensor)
                        self.cache_manager.get(key)
                        self.cache_manager.delete(key)
                        
                    elif op == "model":
                        # Model operation
                        model = torch.nn.Linear(10, 1)
                        path = os.path.join(
                            self.model_dir,
                            f"stress_{time.time()}.pt"
                        )
                        torch.save(model.state_dict(), path)
                        
                    elif op == "knowledge":
                        # Knowledge base operation
                        content = f"Stress test document {time.time()}"
                        asyncio.run(
                            self.knowledge_manager.process_document(
                                "stress_kb",
                                f"stress_{time.time()}.txt",
                                content.encode()
                            )
                        )
                        
                    else:
                        # System operation
                        self.system_manager.get_system_metrics()
                        
                    # Record metrics
                    latency = time.time() - start
                    self.metrics.add_latency(latency)
                    
                return True
                
            except Exception as e:
                self.log_error(e)
                return False
                
        # Create thread pool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:
            # Run stress test
            futures = [
                executor.submit(stress_worker)
                for _ in range(num_threads)
            ]
            
            # Wait for completion
            results = [f.result() for f in futures]
            
        # Calculate metrics
        success_rate = sum(results) / len(results)
        self.metrics.add_error_rate(1 - success_rate)
        
        # Get final resource usage
        system_metrics = self.system_manager.get_system_metrics()
        self.metrics.add_memory_usage(system_metrics["memory_percent"])
        self.metrics.add_cpu_usage(system_metrics["cpu_percent"])
        
        self.metrics.stop()
        
        # Log results
        self.log_event(
            "stress_test",
            self.metrics.get_summary()
        )
        
        # Verify performance
        summary = self.metrics.get_summary()
        self.assertLess(summary["latency"]["p95"], 5.0)  # 95% under 5s
        self.assertGreater(success_rate, 0.90)  # 90% success
        self.assertLess(summary["memory_mb"]["peak"], 1024)  # Under 1GB
        self.assertLess(summary["cpu_percent"]["mean"], 80)  # Under 80% CPU

if __name__ == "__main__":
    unittest.main()