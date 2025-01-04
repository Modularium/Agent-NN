from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import numpy as np
from datetime import datetime, timedelta
import asyncio
import aioredis
import json
import hashlib
from utils.logging_util import LoggerMixin

class PerformanceManager(LoggerMixin):
    """Manager for performance optimization and caching."""
    
    def __init__(self,
                 redis_url: str = "redis://localhost",
                 max_batch_size: int = 32,
                 cache_ttl: int = 3600,
                 max_workers: int = 4):
        """Initialize performance manager.
        
        Args:
            redis_url: Redis connection URL
            max_batch_size: Maximum batch size
            cache_ttl: Cache TTL in seconds
            max_workers: Maximum worker processes
        """
        super().__init__()
        self.redis_url = redis_url
        self.max_batch_size = max_batch_size
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers
        
        # Initialize Redis client
        self.redis = None
        
        # Batch processing queues
        self.batch_queues: Dict[str, asyncio.Queue] = {}
        self.batch_processors: Dict[str, asyncio.Task] = {}
        
        # Load balancing
        self.worker_loads: Dict[str, float] = {}
        self.worker_tasks: Dict[str, List[str]] = {}
        
        # Performance metrics
        self.metrics: Dict[str, List[float]] = {
            "inference_time": [],
            "batch_size": [],
            "cache_hits": [],
            "worker_utilization": []
        }
        
    async def initialize(self):
        """Initialize connections."""
        if not self.redis:
            self.redis = await aioredis.from_url(self.redis_url)
            
    async def cleanup(self):
        """Clean up resources."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            
        # Stop batch processors
        for processor in self.batch_processors.values():
            processor.cancel()
            
    async def get_cached_result(self,
                              model_id: str,
                              inputs: Union[str, Dict[str, Any]]) -> Optional[Any]:
        """Get cached inference result.
        
        Args:
            model_id: Model identifier
            inputs: Model inputs
            
        Returns:
            Optional[Any]: Cached result or None
        """
        await self.initialize()
        
        # Generate cache key
        cache_key = self._generate_cache_key(model_id, inputs)
        
        # Try to get from cache
        cached = await self.redis.get(cache_key)
        if cached:
            self.metrics["cache_hits"].append(1)
            
            # Log cache hit
            self.log_event(
                "cache_hit",
                {
                    "model_id": model_id,
                    "cache_key": cache_key
                }
            )
            
            return json.loads(cached)
            
        self.metrics["cache_hits"].append(0)
        return None
        
    async def cache_result(self,
                         model_id: str,
                         inputs: Union[str, Dict[str, Any]],
                         result: Any):
        """Cache inference result.
        
        Args:
            model_id: Model identifier
            inputs: Model inputs
            result: Result to cache
        """
        await self.initialize()
        
        # Generate cache key
        cache_key = self._generate_cache_key(model_id, inputs)
        
        # Store in cache
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(result)
        )
        
        # Log cache update
        self.log_event(
            "cache_update",
            {
                "model_id": model_id,
                "cache_key": cache_key
            }
        )
        
    def _generate_cache_key(self,
                          model_id: str,
                          inputs: Union[str, Dict[str, Any]]) -> str:
        """Generate cache key for inputs.
        
        Args:
            model_id: Model identifier
            inputs: Model inputs
            
        Returns:
            str: Cache key
        """
        # Convert inputs to string
        if isinstance(inputs, dict):
            input_str = json.dumps(inputs, sort_keys=True)
        else:
            input_str = str(inputs)
            
        # Generate hash
        key = f"{model_id}:{input_str}"
        return hashlib.sha256(key.encode()).hexdigest()
        
    async def add_to_batch(self,
                          model_id: str,
                          inputs: torch.Tensor) -> asyncio.Future:
        """Add inputs to processing batch.
        
        Args:
            model_id: Model identifier
            inputs: Input tensor
            
        Returns:
            asyncio.Future: Future for batch result
        """
        # Create queue if needed
        if model_id not in self.batch_queues:
            self.batch_queues[model_id] = asyncio.Queue()
            self.batch_processors[model_id] = asyncio.create_task(
                self._process_batches(model_id)
            )
            
        # Create future for result
        future = asyncio.Future()
        
        # Add to queue
        await self.batch_queues[model_id].put((inputs, future))
        
        return future
        
    async def _process_batches(self, model_id: str):
        """Process batches for model.
        
        Args:
            model_id: Model identifier
        """
        queue = self.batch_queues[model_id]
        batch: List[Tuple[torch.Tensor, asyncio.Future]] = []
        
        while True:
            try:
                # Get first item
                inputs, future = await queue.get()
                batch.append((inputs, future))
                
                # Try to fill batch
                try:
                    while len(batch) < self.max_batch_size:
                        inputs, future = await asyncio.wait_for(
                            queue.get(),
                            timeout=0.1
                        )
                        batch.append((inputs, future))
                except asyncio.TimeoutError:
                    pass
                    
                # Process batch
                start_time = datetime.now()
                batch_inputs = torch.stack([x[0] for x in batch])
                results = await self._run_batch_inference(model_id, batch_inputs)
                
                # Set futures
                for (_, future), result in zip(batch, results):
                    future.set_result(result)
                    
                # Update metrics
                duration = (datetime.now() - start_time).total_seconds()
                self.metrics["inference_time"].append(duration)
                self.metrics["batch_size"].append(len(batch))
                
                # Log batch processing
                self.log_event(
                    "batch_processed",
                    {
                        "model_id": model_id,
                        "batch_size": len(batch),
                        "duration": duration
                    }
                )
                
                # Clear batch
                batch = []
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                # Log error
                self.log_error(e, {
                    "model_id": model_id,
                    "batch_size": len(batch)
                })
                
                # Set error for futures
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)
                        
                batch = []
                
    async def _run_batch_inference(self,
                                 model_id: str,
                                 inputs: torch.Tensor) -> List[Any]:
        """Run inference on batch.
        
        Args:
            model_id: Model identifier
            inputs: Batch inputs
            
        Returns:
            List[Any]: Batch results
        """
        # Select worker
        worker_id = self._select_worker()
        
        try:
            # Update worker load
            self.worker_loads[worker_id] += len(inputs)
            if model_id not in self.worker_tasks[worker_id]:
                self.worker_tasks[worker_id].append(model_id)
                
            # Run inference
            # This is a placeholder - actual inference would be implemented
            # based on the specific model and framework being used
            results = [
                torch.randn(64)  # Example output
                for _ in range(len(inputs))
            ]
            
            return results
            
        finally:
            # Update worker load
            self.worker_loads[worker_id] -= len(inputs)
            if self.worker_loads[worker_id] == 0:
                self.worker_tasks[worker_id].remove(model_id)
                
    def _select_worker(self) -> str:
        """Select worker for task.
        
        Returns:
            str: Worker identifier
        """
        # Initialize workers if needed
        if not self.worker_loads:
            for i in range(self.max_workers):
                worker_id = f"worker_{i}"
                self.worker_loads[worker_id] = 0
                self.worker_tasks[worker_id] = []
                
        # Select worker with lowest load
        return min(
            self.worker_loads.items(),
            key=lambda x: x[1]
        )[0]
        
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics.
        
        Returns:
            Dict[str, Dict[str, float]]: Performance metrics
        """
        metrics = {}
        
        for name, values in self.metrics.items():
            if values:
                metrics[name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                
        # Add worker metrics
        metrics["worker_load"] = {
            worker_id: load
            for worker_id, load in self.worker_loads.items()
        }
        
        metrics["worker_tasks"] = {
            worker_id: len(tasks)
            for worker_id, tasks in self.worker_tasks.items()
        }
        
        return metrics
        
    async def optimize_batch_size(self,
                                model_id: str,
                                min_size: int = 1,
                                max_size: int = 128,
                                num_trials: int = 10) -> int:
        """Optimize batch size for model.
        
        Args:
            model_id: Model identifier
            min_size: Minimum batch size
            max_size: Maximum batch size
            num_trials: Number of trials
            
        Returns:
            int: Optimal batch size
        """
        best_size = self.max_batch_size
        best_throughput = 0
        
        for size in np.linspace(min_size, max_size, num_trials, dtype=int):
            # Test batch size
            self.max_batch_size = int(size)
            start_time = datetime.now()
            
            # Run test batches
            futures = []
            for _ in range(100):
                inputs = torch.randn(64)  # Example input
                futures.append(await self.add_to_batch(model_id, inputs))
                
            # Wait for completion
            await asyncio.gather(*futures)
            
            # Calculate throughput
            duration = (datetime.now() - start_time).total_seconds()
            throughput = len(futures) / duration
            
            # Update best size
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = size
                
            # Log trial
            self.log_event(
                "batch_size_trial",
                {
                    "model_id": model_id,
                    "batch_size": size,
                    "throughput": throughput
                }
            )
            
        # Restore best size
        self.max_batch_size = int(best_size)
        return best_size