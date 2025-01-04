import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from langchain.schema import Document
from rag.url_rag_system import URLRAGSystem

class TestURLRAGSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Mock MLflow
        self.mlflow_patcher = patch('utils.logging_util.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        self.mock_mlflow.active_run.return_value = None
        
        # Mock agents
        self.scraper_patcher = patch('rag.url_rag_system.WebScraperAgent')
        self.crawler_patcher = patch('rag.url_rag_system.WebCrawlerAgent')
        self.vector_store_patcher = patch('rag.url_rag_system.VectorStore')
        
        self.mock_scraper = self.scraper_patcher.start()
        self.mock_crawler = self.crawler_patcher.start()
        self.mock_vector_store = self.vector_store_patcher.start()
        
        # Set up mock instances
        self.mock_scraper_instance = AsyncMock()
        self.mock_crawler_instance = AsyncMock()
        self.mock_vector_store_instance = MagicMock()
        
        self.mock_scraper.return_value = self.mock_scraper_instance
        self.mock_crawler.return_value = self.mock_crawler_instance
        self.mock_vector_store.return_value = self.mock_vector_store_instance
        
        # Initialize system
        self.rag = URLRAGSystem(
            name="test_rag",
            update_interval=timedelta(days=1)
        )
        
        # Sample data
        self.sample_urls = ["http://example.com/1", "http://example.com/2"]
        self.sample_selectors = {
            "title": "h1",
            "content": "article"
        }
        
    def tearDown(self):
        """Clean up after tests."""
        self.mlflow_patcher.stop()
        self.scraper_patcher.stop()
        self.crawler_patcher.stop()
        self.vector_store_patcher.stop()
        
    async def test_scrape_and_index(self):
        """Test scraping and indexing content."""
        # Mock scraper results
        self.mock_scraper_instance.scrape_multiple_urls.return_value = [
            {
                "title": "Test Title 1",
                "content": "Test Content 1",
                "source_url": self.sample_urls[0],
                "timestamp": datetime.now().isoformat()
            },
            {
                "title": "Test Title 2",
                "content": "Test Content 2",
                "source_url": self.sample_urls[1],
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Test scraping
        await self.rag.scrape_and_index(
            self.sample_urls,
            self.sample_selectors
        )
        
        # Check scraper was called
        self.mock_scraper_instance.scrape_multiple_urls.assert_called_once_with(
            self.sample_urls,
            self.sample_selectors,
            None
        )
        
        # Check documents were added to vector store
        self.mock_vector_store_instance.add_documents.assert_called()
        
        # Check URLs were tracked
        self.assertEqual(len(self.rag.processed_urls), 2)
        
    async def test_crawl_and_index(self):
        """Test crawling and indexing content."""
        # Mock crawler results
        self.mock_crawler_instance.crawl.return_value = [
            {
                "url": self.sample_urls[0],
                "title": "Test Title 1",
                "description": "Test Description 1",
                "content": "Test Content 1",
                "timestamp": datetime.now().isoformat(),
                "depth": 0
            },
            {
                "url": self.sample_urls[1],
                "title": "Test Title 2",
                "description": "Test Description 2",
                "content": "Test Content 2",
                "timestamp": datetime.now().isoformat(),
                "depth": 1
            }
        ]
        
        # Test crawling
        await self.rag.crawl_and_index(self.sample_urls)
        
        # Check crawler was called
        self.mock_crawler_instance.crawl.assert_called_once_with(
            self.sample_urls
        )
        
        # Check documents were added to vector store
        self.mock_vector_store_instance.add_documents.assert_called()
        
        # Check URLs were tracked
        self.assertEqual(len(self.rag.processed_urls), 2)
        
    async def test_update_content(self):
        """Test content updating."""
        # Add some processed URLs
        self.rag.processed_urls = {
            self.sample_urls[0]: datetime.now() - timedelta(days=2),
            self.sample_urls[1]: datetime.now()
        }
        
        # Mock vector store search
        self.mock_vector_store_instance.similarity_search.return_value = [
            Document(
                page_content="test",
                metadata={"source": "scraper", "url": self.sample_urls[0]}
            )
        ]
        
        # Test update
        await self.rag.update_content()
        
        # Check that only old URL was updated
        self.mock_scraper_instance.scrape_multiple_urls.assert_called_once()
        self.assertEqual(
            len(self.mock_scraper_instance.scrape_multiple_urls.call_args[0][0]),
            1
        )
        
    def test_search(self):
        """Test knowledge base search."""
        # Mock search results
        mock_docs = [
            Document(
                page_content="test content",
                metadata={"url": "http://example.com"}
            )
        ]
        self.mock_vector_store_instance.similarity_search.return_value = mock_docs
        
        # Test search
        results = self.rag.search("test query", k=1)
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "test content")
        
    def test_get_source_stats(self):
        """Test source statistics generation."""
        # Mock search results
        mock_docs = [
            Document(
                page_content="test1",
                metadata={
                    "source": "scraper",
                    "url": "http://example.com/1",
                    "timestamp": datetime.now().isoformat()
                }
            ),
            Document(
                page_content="test2",
                metadata={
                    "source": "crawler",
                    "url": "http://example.com/2",
                    "timestamp": datetime.now().isoformat(),
                    "depth": 1
                }
            )
        ]
        self.mock_vector_store_instance.similarity_search.return_value = mock_docs
        
        # Get stats
        stats = self.rag.get_source_stats()
        
        # Check stats
        self.assertEqual(len(stats), 2)  # Two sources
        self.assertEqual(stats.index.tolist(), ["crawler", "scraper"])
        
    async def test_cleanup_old_content(self):
        """Test old content cleanup."""
        # Mock search results with old and new documents
        old_time = datetime.now() - timedelta(days=40)
        new_time = datetime.now() - timedelta(days=1)
        
        mock_docs = [
            Document(
                page_content="old",
                metadata={
                    "url": "http://example.com/old",
                    "timestamp": old_time.isoformat()
                }
            ),
            Document(
                page_content="new",
                metadata={
                    "url": "http://example.com/new",
                    "timestamp": new_time.isoformat()
                }
            )
        ]
        self.mock_vector_store_instance.similarity_search.return_value = mock_docs
        
        # Add to processed URLs
        self.rag.processed_urls = {
            "http://example.com/old": old_time,
            "http://example.com/new": new_time
        }
        
        # Test cleanup
        await self.rag.cleanup_old_content(max_age=timedelta(days=30))
        
        # Check that old document was removed
        self.mock_vector_store_instance.delete_documents.assert_called_once_with(
            ["http://example.com/old"]
        )
        
        # Check processed URLs were updated
        self.assertEqual(len(self.rag.processed_urls), 1)
        self.assertIn("http://example.com/new", self.rag.processed_urls)

if __name__ == '__main__':
    unittest.main()