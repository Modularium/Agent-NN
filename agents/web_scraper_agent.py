import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urljoin, urlparse
from datetime import datetime
import time
from utils.logging_util import LoggerMixin
from datastores.worker_agent_db import WorkerAgentDB
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class WebScraperAgent(LoggerMixin):
    """Agent for extracting specific data from websites."""
    
    def __init__(self,
                 name: str = "web_scraper",
                 rate_limit: float = 1.0,
                 respect_robots: bool = True,
                 max_retries: int = 3,
                 timeout: float = 30.0):
        """Initialize web scraper agent.
        
        Args:
            name: Agent name
            rate_limit: Minimum seconds between requests to same domain
            respect_robots: Whether to respect robots.txt
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.name = name
        self.rate_limit = rate_limit
        self.respect_robots = respect_robots
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize database
        self.db = WorkerAgentDB(name)
        
        # Rate limiting state
        self.last_request: Dict[str, float] = {}
        
        # Initialize text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Session for making requests
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def scrape_url(self,
                        url: str,
                        selectors: Dict[str, str],
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Scrape data from a URL using CSS selectors.
        
        Args:
            url: URL to scrape
            selectors: Dictionary of field names to CSS selectors
            metadata: Optional metadata to store with results
            
        Returns:
            Dict[str, Any]: Extracted data
        """
        await self.initialize()
        
        # Check rate limiting
        domain = urlparse(url).netloc
        if domain in self.last_request:
            time_since_last = time.time() - self.last_request[domain]
            if time_since_last < self.rate_limit:
                await asyncio.sleep(self.rate_limit - time_since_last)
                
        # Update last request time
        self.last_request[domain] = time.time()
        
        # Make request with retries
        content = None
        errors = []
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        content = await response.text()
                        break
                    else:
                        errors.append(f"HTTP {response.status}")
            except Exception as e:
                errors.append(str(e))
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        if not content:
            error_msg = f"Failed to fetch {url} after {self.max_retries} attempts: {errors}"
            self.log_error(Exception(error_msg), {"url": url})
            return {}
            
        # Parse content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract data using selectors
        results = {}
        for field, selector in selectors.items():
            elements = soup.select(selector)
            if elements:
                # Handle different types of data
                if len(elements) == 1:
                    results[field] = elements[0].get_text(strip=True)
                else:
                    results[field] = [el.get_text(strip=True) for el in elements]
                    
        # Add metadata
        if metadata:
            results.update(metadata)
            
        # Add timestamp and URL
        results["timestamp"] = datetime.now().isoformat()
        results["source_url"] = url
        
        # Log success
        self.log_event(
            "url_scraped",
            {
                "url": url,
                "fields": list(selectors.keys()),
                "num_results": len(results)
            }
        )
        
        return results
        
    async def scrape_multiple_urls(self,
                                 urls: List[str],
                                 selectors: Dict[str, str],
                                 metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Scrape data from multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            selectors: Dictionary of field names to CSS selectors
            metadata: Optional metadata to store with results
            
        Returns:
            List[Dict[str, Any]]: List of extracted data
        """
        tasks = [
            self.scrape_url(url, selectors, metadata)
            for url in urls
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Filter out empty results
        return [r for r in results if r]
        
    def store_results(self,
                     results: List[Dict[str, Any]],
                     format: str = "documents") -> None:
        """Store scraped results in the database.
        
        Args:
            results: List of scraped data
            format: Storage format ("documents" or "json")
        """
        if format == "documents":
            # Convert to documents
            documents = []
            for result in results:
                # Create content string
                content = "\n".join(
                    f"{k}: {v}"
                    for k, v in result.items()
                    if k not in ["timestamp", "source_url"]
                )
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata={
                        "timestamp": result["timestamp"],
                        "source_url": result["source_url"]
                    }
                )
                documents.append(doc)
                
            # Split into chunks if needed
            if documents:
                chunks = self.text_splitter.split_documents(documents)
                self.db.ingest_documents(chunks)
                
        elif format == "json":
            # Store as JSON
            self.db.store_json(results)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Log storage
        self.log_event(
            "results_stored",
            {
                "format": format,
                "num_results": len(results)
            }
        )
        
    def search_stored_data(self,
                          query: str,
                          k: int = 4) -> List[Document]:
        """Search stored documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Document]: Matching documents
        """
        return self.db.search(query, k=k)
        
    def get_stored_json(self,
                       query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get stored JSON data.
        
        Args:
            query: Optional MongoDB-style query
            
        Returns:
            List[Dict[str, Any]]: Matching JSON documents
        """
        return self.db.get_json(query)
        
    def export_to_csv(self,
                     results: List[Dict[str, Any]],
                     output_path: str) -> None:
        """Export results to CSV file.
        
        Args:
            results: List of results to export
            output_path: Path to output CSV file
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        
        self.log_event(
            "results_exported",
            {
                "format": "csv",
                "path": output_path,
                "num_rows": len(df)
            }
        )