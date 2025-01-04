import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime, timedelta
import aioredis
from rag.content_cache import ContentCache
from rag.js_renderer import JSRenderer
from rag.parallel_processor import ParallelProcessor, BatchStats

class TestContentCache(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock Redis
        self.redis_patcher = patch('rag.content_cache.aioredis')
        self.mock_redis = self.redis_patcher.start()
        
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock Redis instance
        self.mock_redis_instance = AsyncMock()
        self.mock_redis.from_url.return_value = self.mock_redis_instance
        
        # Initialize cache
        self.cache = ContentCache(
            redis_url="redis://test",
            ttl=3600,
            max_size=1000
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.redis_patcher.stop()
        self.mlflow_patcher.stop()
        
    async def test_cache_operations(self):
        """Test basic cache operations."""
        # Test cache set
        content = {"test": "data"}
        success = await self.cache.set("http://test.com", content)
        self.assertTrue(success)
        
        # Verify Redis call
        self.mock_redis_instance.setex.assert_called_once()
        
        # Test cache get
        self.mock_redis_instance.get.return_value = '{"test": "data"}'
        result = await self.cache.get("http://test.com")
        self.assertEqual(result, content)
        
        # Test cache miss
        self.mock_redis_instance.get.return_value = None
        result = await self.cache.get("http://missing.com")
        self.assertIsNone(result)
        
    async def test_cache_eviction(self):
        """Test cache eviction."""
        # Mock cache size
        self.mock_redis_instance.dbsize.return_value = 1000
        
        # Add item that triggers eviction
        await self.cache.set("http://test.com", {"test": "data"})
        
        # Verify eviction
        self.mock_redis_instance.delete.assert_called()

class TestJSRenderer(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock Playwright
        self.playwright_patcher = patch('rag.js_renderer.async_playwright')
        self.mock_playwright = self.playwright_patcher.start()
        
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Set up mock browser and page
        self.mock_browser = AsyncMock()
        self.mock_page = AsyncMock()
        self.mock_browser.new_page.return_value = self.mock_page
        
        # Set up mock Playwright instance
        mock_playwright_instance = AsyncMock()
        mock_playwright_instance.chromium.launch.return_value = self.mock_browser
        self.mock_playwright.return_value = mock_playwright_instance
        
        # Initialize renderer
        self.renderer = JSRenderer(
            headless=True,
            timeout=30000
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.playwright_patcher.stop()
        self.mlflow_patcher.stop()
        
    async def test_page_rendering(self):
        """Test page rendering."""
        # Mock page content
        self.mock_page.content.return_value = "<html>test</html>"
        self.mock_page.evaluate.return_value = "test content"
        
        # Render page
        content = await self.renderer.render_page(
            "http://test.com",
            selectors={"title": "h1"}
        )
        
        # Verify navigation
        self.mock_page.goto.assert_called_once_with(
            "http://test.com",
            wait_until="networkidle",
            timeout=30000
        )
        
        # Verify content extraction
        self.assertIn("title", content)
        
    async def test_error_handling(self):
        """Test error handling and retries."""
        # Make first attempt fail
        self.mock_page.goto.side_effect = [
            Exception("Navigation failed"),
            None  # Second attempt succeeds
        ]
        
        # Render page
        await self.renderer.render_page("http://test.com")
        
        # Verify retry
        self.assertEqual(self.mock_page.goto.call_count, 2)
        
        # Check error stats
        stats = await self.renderer.get_error_stats()
        self.assertEqual(stats["total_errors"], 1)

class TestParallelProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Initialize processor
        self.processor = ParallelProcessor(
            max_concurrency=5,
            batch_size=10
        )
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        
    async def test_parallel_processing(self):
        """Test parallel item processing."""
        # Create test items
        items = list(range(25))
        
        # Create mock processor
        async def mock_process(item):
            await asyncio.sleep(0.1)  # Simulate work
            return item * 2
            
        # Process items
        results = await self.processor.process_items(items, mock_process)
        
        # Verify results
        self.assertEqual(len(results), 25)
        self.assertEqual(results, [i * 2 for i in items])
        
        # Check stats
        stats = self.processor.get_processing_stats()
        self.assertEqual(stats["total_successful"], 25)
        self.assertEqual(stats["total_failed"], 0)
        
    async def test_error_handling(self):
        """Test error handling in parallel processing."""
        # Create test items
        items = list(range(10))
        
        # Create mock processor that fails sometimes
        async def mock_process(item):
            if item % 2 == 0:
                raise Exception(f"Error processing {item}")
            return item
            
        # Create mock error handler
        error_handler = AsyncMock()
        
        # Process items
        results = await self.processor.process_items(
            items,
            mock_process,
            error_handler
        )
        
        # Verify results
        self.assertEqual(len(results), 5)  # Only odd numbers succeed
        
        # Verify error handling
        self.assertEqual(error_handler.call_count, 5)  # Called for even numbers
        
        # Check stats
        stats = self.processor.get_processing_stats()
        self.assertEqual(stats["total_successful"], 5)
        self.assertEqual(stats["total_failed"], 5)

if __name__ == '__main__':
    unittest.main()