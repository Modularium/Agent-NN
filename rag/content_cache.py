from typing import Dict, Any, Optional, List
import time
from datetime import datetime, timedelta
import json
import hashlib
import aioredis
import asyncio
from utils.logging_util import LoggerMixin

class ContentCache(LoggerMixin):
    """Cache for frequently accessed web content."""
    
    def __init__(self,
                 redis_url: str = "redis://localhost",
                 ttl: int = 3600,  # 1 hour default TTL
                 max_size: int = 10000):
        """Initialize content cache.
        
        Args:
            redis_url: Redis connection URL
            ttl: Time to live in seconds
            max_size: Maximum number of items in cache
        """
        super().__init__()
        self.redis_url = redis_url
        self.ttl = ttl
        self.max_size = max_size
        self.redis: Optional[aioredis.Redis] = None
        
        # Access frequency tracking
        self.access_counts: Dict[str, int] = {}
        
    async def initialize(self):
        """Initialize Redis connection."""
        if not self.redis:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
    async def cleanup(self):
        """Clean up resources."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            
    def _get_key(self, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from URL and parameters.
        
        Args:
            url: URL to cache
            params: Optional parameters
            
        Returns:
            str: Cache key
        """
        key = url
        if params:
            key += "_" + json.dumps(params, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()
        
    async def get(self,
                 url: str,
                 params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get content from cache.
        
        Args:
            url: URL to retrieve
            params: Optional parameters
            
        Returns:
            Optional[Dict[str, Any]]: Cached content or None
        """
        await self.initialize()
        
        key = self._get_key(url, params)
        try:
            # Get cached data
            data = await self.redis.get(key)
            if data:
                # Update access count
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                # Parse data
                content = json.loads(data)
                
                # Log cache hit
                self.log_event(
                    "cache_hit",
                    {
                        "url": url,
                        "access_count": self.access_counts[key]
                    }
                )
                
                return content
                
        except Exception as e:
            self.log_error(e, {
                "url": url,
                "operation": "cache_get"
            })
            
        return None
        
    async def set(self,
                  url: str,
                  content: Dict[str, Any],
                  params: Optional[Dict[str, Any]] = None) -> bool:
        """Store content in cache.
        
        Args:
            url: URL to cache
            content: Content to cache
            params: Optional parameters
            
        Returns:
            bool: Whether content was cached
        """
        await self.initialize()
        
        key = self._get_key(url, params)
        try:
            # Check cache size
            if await self.redis.dbsize() >= self.max_size:
                await self._evict_items()
                
            # Store content
            data = json.dumps(content)
            await self.redis.setex(key, self.ttl, data)
            
            # Initialize access count
            self.access_counts[key] = 0
            
            # Log cache set
            self.log_event(
                "cache_set",
                {
                    "url": url,
                    "content_size": len(data)
                }
            )
            
            return True
            
        except Exception as e:
            self.log_error(e, {
                "url": url,
                "operation": "cache_set"
            })
            
        return False
        
    async def _evict_items(self, count: int = 100):
        """Evict least frequently accessed items.
        
        Args:
            count: Number of items to evict
        """
        # Sort by access count
        items = sorted(
            self.access_counts.items(),
            key=lambda x: x[1]
        )
        
        # Remove least accessed items
        for key, _ in items[:count]:
            await self.redis.delete(key)
            del self.access_counts[key]
            
        # Log eviction
        self.log_event(
            "cache_eviction",
            {"items_evicted": count}
        )
        
    async def clear(self):
        """Clear all cached content."""
        await self.initialize()
        
        try:
            await self.redis.flushdb()
            self.access_counts.clear()
            
            # Log clear
            self.log_event("cache_cleared", {})
            
        except Exception as e:
            self.log_error(e, {"operation": "cache_clear"})
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        await self.initialize()
        
        try:
            size = await self.redis.dbsize()
            keys = await self.redis.keys("*")
            
            # Calculate hit rates
            total_accesses = sum(self.access_counts.values())
            hits = len([c for c in self.access_counts.values() if c > 0])
            
            stats = {
                "size": size,
                "max_size": self.max_size,
                "utilization": size / self.max_size,
                "total_accesses": total_accesses,
                "hit_rate": hits / len(keys) if keys else 0,
                "most_accessed": sorted(
                    self.access_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
            
            # Log stats
            self.log_event("cache_stats", stats)
            
            return stats
            
        except Exception as e:
            self.log_error(e, {"operation": "get_stats"})
            return {}