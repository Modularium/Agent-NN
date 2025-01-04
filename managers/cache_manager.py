from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
import torch
import numpy as np
from collections import OrderedDict
import threading
import time
from datetime import datetime, timedelta
import json
import os
import mlflow
from utils.logging_util import LoggerMixin

T = TypeVar('T')

class CachePolicy:
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live

class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    
    def __init__(self,
                 key: str,
                 value: T,
                 ttl: Optional[int] = None):
        """Initialize entry.
        
        Args:
            key: Cache key
            value: Cached value
            ttl: Time to live in seconds
        """
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.ttl = ttl
        
    def access(self):
        """Record cache access."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    def is_expired(self) -> bool:
        """Check if entry is expired.
        
        Returns:
            bool: Whether entry is expired
        """
        if not self.ttl:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl

class CacheManager(LoggerMixin):
    """Manager for caching system."""
    
    def __init__(self,
                 max_size: int = 1024,  # MB
                 policy: str = CachePolicy.LRU,
                 default_ttl: Optional[int] = None,
                 cleanup_interval: int = 60):  # seconds
        """Initialize manager.
        
        Args:
            max_size: Maximum cache size in MB
            policy: Cache eviction policy
            default_ttl: Default time to live in seconds
            cleanup_interval: Cache cleanup interval
        """
        super().__init__()
        self.max_size = max_size * 1024 * 1024  # Convert to bytes
        self.policy = policy
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Initialize caches
        self.memory_cache: Dict[str, CacheEntry] = OrderedDict()
        self.tensor_cache: Dict[str, CacheEntry[torch.Tensor]] = OrderedDict()
        self.embedding_cache: Dict[str, CacheEntry[np.ndarray]] = OrderedDict()
        
        # Initialize lock
        self.lock = threading.Lock()
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("cache_management")
        
        # Start cleanup thread
        self.running = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop
        )
        self.cleanup_thread.start()
        
    def _get_size(self, value: Any) -> int:
        """Get value size in bytes.
        
        Args:
            value: Value to measure
            
        Returns:
            int: Size in bytes
        """
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, torch.Tensor):
            return value.element_size() * value.nelement()
        elif isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (dict, list)):
            return len(json.dumps(value).encode())
        return 0
        
    def _get_total_size(self) -> int:
        """Get total cache size.
        
        Returns:
            int: Total size in bytes
        """
        total = 0
        for cache in [
            self.memory_cache,
            self.tensor_cache,
            self.embedding_cache
        ]:
            for entry in cache.values():
                total += self._get_size(entry.value)
        return total
        
    def _evict_entries(self, required_size: int):
        """Evict cache entries.
        
        Args:
            required_size: Required size in bytes
        """
        if self.policy == CachePolicy.LRU:
            # Remove least recently used entries
            for cache in [
                self.memory_cache,
                self.tensor_cache,
                self.embedding_cache
            ]:
                while (cache and
                       self._get_total_size() + required_size > self.max_size):
                    # Remove oldest entry
                    key = next(iter(cache))
                    del cache[key]
                    
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used entries
            entries = []
            for cache in [
                self.memory_cache,
                self.tensor_cache,
                self.embedding_cache
            ]:
                entries.extend([
                    (key, entry)
                    for key, entry in cache.items()
                ])
                
            # Sort by access count
            entries.sort(key=lambda x: x[1].access_count)
            
            # Remove entries
            while (entries and
                   self._get_total_size() + required_size > self.max_size):
                key, entry = entries.pop(0)
                if key in self.memory_cache:
                    del self.memory_cache[key]
                elif key in self.tensor_cache:
                    del self.tensor_cache[key]
                elif key in self.embedding_cache:
                    del self.embedding_cache[key]
                    
    def _cleanup_loop(self):
        """Cache cleanup loop."""
        while self.running:
            try:
                # Remove expired entries
                with self.lock:
                    for cache in [
                        self.memory_cache,
                        self.tensor_cache,
                        self.embedding_cache
                    ]:
                        expired = [
                            key
                            for key, entry in cache.items()
                            if entry.is_expired()
                        ]
                        for key in expired:
                            del cache[key]
                            
                # Log cleanup
                if expired:
                    self.log_event(
                        "cache_cleanup",
                        {"expired": len(expired)}
                    )
                    
            except Exception as e:
                self.log_error(e)
                
            time.sleep(self.cleanup_interval)
            
    def set(self,
            key: str,
            value: T,
            ttl: Optional[int] = None) -> bool:
        """Set cache entry.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time to live
            
        Returns:
            bool: Whether value was cached
        """
        try:
            with self.lock:
                # Check size
                size = self._get_size(value)
                if size > self.max_size:
                    return False
                    
                # Evict entries if needed
                self._evict_entries(size)
                
                # Create entry
                entry = CacheEntry(
                    key,
                    value,
                    ttl or self.default_ttl
                )
                
                # Add to appropriate cache
                if isinstance(value, torch.Tensor):
                    self.tensor_cache[key] = entry
                elif isinstance(value, np.ndarray):
                    self.embedding_cache[key] = entry
                else:
                    self.memory_cache[key] = entry
                    
                return True
                
        except Exception as e:
            self.log_error(e, {"key": key})
            return False
            
    def get(self, key: str) -> Optional[T]:
        """Get cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[T]: Cached value
        """
        try:
            with self.lock:
                # Check caches
                entry = None
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                elif key in self.tensor_cache:
                    entry = self.tensor_cache[key]
                elif key in self.embedding_cache:
                    entry = self.embedding_cache[key]
                    
                if not entry:
                    return None
                    
                # Check expiration
                if entry.is_expired():
                    self.delete(key)
                    return None
                    
                # Update access
                entry.access()
                return entry.value
                
        except Exception as e:
            self.log_error(e, {"key": key})
            return None
            
    def delete(self, key: str):
        """Delete cache entry.
        
        Args:
            key: Cache key
        """
        with self.lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
            elif key in self.tensor_cache:
                del self.tensor_cache[key]
            elif key in self.embedding_cache:
                del self.embedding_cache[key]
                
    def clear(self):
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            self.tensor_cache.clear()
            self.embedding_cache.clear()
            
            # Log clear
            self.log_event("cache_cleared", {})
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self.lock:
            # Calculate sizes
            memory_size = 0
            for entry in self.memory_cache.values():
                try:
                    memory_size += self._get_size(entry.value)
                except Exception:
                    pass
                    
            tensor_size = 0
            for entry in self.tensor_cache.values():
                try:
                    tensor_size += self._get_size(entry.value)
                except Exception:
                    pass
                    
            embedding_size = 0
            for entry in self.embedding_cache.values():
                try:
                    embedding_size += self._get_size(entry.value)
                except Exception:
                    pass
            
            return {
                "total_size": memory_size + tensor_size + embedding_size,
                "max_size": self.max_size,
                "memory_entries": len(self.memory_cache),
                "memory_size": memory_size,
                "tensor_entries": len(self.tensor_cache),
                "tensor_size": tensor_size,
                "embedding_entries": len(self.embedding_cache),
                "embedding_size": embedding_size,
                "policy": self.policy
            }
            
    def stop(self):
        """Stop cache manager."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
            
    def __del__(self):
        """Clean up resources."""
        self.stop()