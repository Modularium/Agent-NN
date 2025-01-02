from typing import List, Dict, Any, Optional, Callable, Awaitable, TypeVar
import asyncio
from datetime import datetime
import time
import math
from dataclasses import dataclass
from utils.logging_util import LoggerMixin

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class BatchStats:
    """Statistics for a processing batch."""
    batch_size: int
    start_time: datetime
    end_time: Optional[datetime] = None
    successful: int = 0
    failed: int = 0
    total_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful + self.failed
        return self.successful / total if total > 0 else 0.0
        
    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        if not self.end_time:
            return 0.0
        duration = (self.end_time - self.start_time).total_seconds()
        return self.successful / duration if duration > 0 else 0.0

class ParallelProcessor(LoggerMixin):
    """Parallel processing manager for RAG system tasks."""
    
    def __init__(self,
                 max_concurrency: int = 10,
                 batch_size: int = 100,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """Initialize parallel processor.
        
        Args:
            max_concurrency: Maximum concurrent tasks
            batch_size: Size of processing batches
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (seconds)
        """
        super().__init__()
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Processing statistics
        self.batch_stats: List[BatchStats] = []
        
        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
    async def process_items(self,
                          items: List[T],
                          processor: Callable[[T], Awaitable[U]],
                          error_handler: Optional[Callable[[T, Exception], Awaitable[None]]] = None) -> List[U]:
        """Process items in parallel with batching.
        
        Args:
            items: Items to process
            processor: Async function to process each item
            error_handler: Optional error handler
            
        Returns:
            List[U]: Processing results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await self._process_batch(
                batch,
                processor,
                error_handler
            )
            results.extend(batch_results)
            
        return results
        
    async def _process_batch(self,
                           batch: List[T],
                           processor: Callable[[T], Awaitable[U]],
                           error_handler: Optional[Callable[[T, Exception], Awaitable[None]]]) -> List[U]:
        """Process a batch of items.
        
        Args:
            batch: Batch of items
            processor: Processing function
            error_handler: Error handler
            
        Returns:
            List[U]: Batch results
        """
        # Initialize batch statistics
        stats = BatchStats(
            batch_size=len(batch),
            start_time=datetime.now()
        )
        
        # Create tasks
        tasks = []
        for item in batch:
            task = asyncio.create_task(
                self._process_item(
                    item,
                    processor,
                    error_handler,
                    stats
                )
            )
            tasks.append(task)
            
        # Wait for all tasks
        results = await asyncio.gather(*tasks)
        
        # Update batch statistics
        stats.end_time = datetime.now()
        stats.total_time = (stats.end_time - stats.start_time).total_seconds()
        self.batch_stats.append(stats)
        
        # Log batch completion
        self.log_event(
            "batch_complete",
            {
                "batch_size": stats.batch_size,
                "successful": stats.successful,
                "failed": stats.failed,
                "success_rate": stats.success_rate,
                "items_per_second": stats.items_per_second
            }
        )
        
        # Filter out None results (failed items)
        return [r for r in results if r is not None]
        
    async def _process_item(self,
                          item: T,
                          processor: Callable[[T], Awaitable[U]],
                          error_handler: Optional[Callable[[T, Exception], Awaitable[None]]],
                          stats: BatchStats) -> Optional[U]:
        """Process a single item with retries.
        
        Args:
            item: Item to process
            processor: Processing function
            error_handler: Error handler
            stats: Batch statistics
            
        Returns:
            Optional[U]: Processing result or None if failed
        """
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    result = await processor(item)
                    stats.successful += 1
                    return result
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        # Final attempt failed
                        stats.failed += 1
                        if error_handler:
                            await error_handler(item, e)
                        self.log_error(e, {
                            "item": str(item),
                            "attempt": attempt + 1
                        })
                        return None
                        
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        if not self.batch_stats:
            return {}
            
        total_items = sum(s.batch_size for s in self.batch_stats)
        total_successful = sum(s.successful for s in self.batch_stats)
        total_failed = sum(s.failed for s in self.batch_stats)
        total_time = sum(s.total_time for s in self.batch_stats)
        
        return {
            "total_batches": len(self.batch_stats),
            "total_items": total_items,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": total_successful / total_items,
            "total_processing_time": total_time,
            "average_batch_time": total_time / len(self.batch_stats),
            "items_per_second": total_successful / total_time if total_time > 0 else 0,
            "batch_stats": [
                {
                    "size": s.batch_size,
                    "successful": s.successful,
                    "failed": s.failed,
                    "success_rate": s.success_rate,
                    "items_per_second": s.items_per_second
                }
                for s in self.batch_stats[-10:]  # Last 10 batches
            ]
        }
        
    def clear_stats(self):
        """Clear processing statistics."""
        self.batch_stats.clear()
        self.log_event("stats_cleared", {})