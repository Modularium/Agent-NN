import asyncio
from typing import List, Dict, Any, Optional, Set
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import re
import time
from datetime import datetime
import hashlib
from utils.logging_util import LoggerMixin
from datastores.worker_agent_db import WorkerAgentDB
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class WebCrawlerAgent(LoggerMixin):
    """Agent for systematically crawling and indexing websites."""
    
    def __init__(self,
                 name: str = "web_crawler",
                 rate_limit: float = 1.0,
                 respect_robots: bool = True,
                 max_depth: int = 3,
                 max_pages: int = 1000,
                 allowed_domains: Optional[List[str]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        """Initialize web crawler agent.
        
        Args:
            name: Agent name
            rate_limit: Minimum seconds between requests to same domain
            respect_robots: Whether to respect robots.txt
            max_depth: Maximum crawl depth from start URL
            max_pages: Maximum number of pages to crawl
            allowed_domains: List of allowed domains (None for any)
            exclude_patterns: List of URL patterns to exclude
        """
        super().__init__()
        self.name = name
        self.rate_limit = rate_limit
        self.respect_robots = respect_robots
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.allowed_domains = set(allowed_domains) if allowed_domains else None
        self.exclude_patterns = [re.compile(p) for p in (exclude_patterns or [])]
        
        # Initialize database
        self.db = WorkerAgentDB(name)
        
        # Rate limiting and crawl state
        self.last_request: Dict[str, float] = {}
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.crawled_urls: Set[str] = set()
        self.url_queue: asyncio.Queue = asyncio.Queue()
        
        # Initialize text splitter
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
            
    async def get_robots_parser(self, domain: str) -> Optional[RobotFileParser]:
        """Get or create robots.txt parser for a domain.
        
        Args:
            domain: Domain to get robots.txt for
            
        Returns:
            Optional[RobotFileParser]: Parser or None if not available
        """
        if domain in self.robots_cache:
            return self.robots_cache[domain]
            
        try:
            robots_url = f"https://{domain}/robots.txt"
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    content = await response.text()
                    parser = RobotFileParser()
                    parser.parse(content.splitlines())
                    self.robots_cache[domain] = parser
                    return parser
        except Exception as e:
            self.log_error(e, {"domain": domain})
            
        return None
        
    def can_crawl_url(self, url: str) -> bool:
        """Check if URL is allowed to be crawled.
        
        Args:
            url: URL to check
            
        Returns:
            bool: Whether URL can be crawled
        """
        parsed = urlparse(url)
        
        # Check domain
        if self.allowed_domains and parsed.netloc not in self.allowed_domains:
            return False
            
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if pattern.search(url):
                return False
                
        return True
        
    async def extract_links(self, url: str, html: str) -> List[str]:
        """Extract links from HTML content.
        
        Args:
            url: Source URL
            html: HTML content
            
        Returns:
            List[str]: List of extracted links
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(url, href)
            
            # Filter URLs
            if (full_url.startswith(('http://', 'https://')) and
                self.can_crawl_url(full_url)):
                links.append(full_url)
                
        return links
        
    async def crawl_url(self,
                       url: str,
                       depth: int) -> Optional[Dict[str, Any]]:
        """Crawl a single URL.
        
        Args:
            url: URL to crawl
            depth: Current crawl depth
            
        Returns:
            Optional[Dict[str, Any]]: Crawled data or None if failed
        """
        # Check if already crawled
        if url in self.crawled_urls:
            return None
            
        # Add to crawled set
        self.crawled_urls.add(url)
        
        # Parse URL
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Check robots.txt
        if self.respect_robots:
            parser = await self.get_robots_parser(domain)
            if parser and not parser.can_fetch("*", url):
                return None
                
        # Check rate limiting
        if domain in self.last_request:
            time_since_last = time.time() - self.last_request[domain]
            if time_since_last < self.rate_limit:
                await asyncio.sleep(self.rate_limit - time_since_last)
                
        # Update last request time
        self.last_request[domain] = time.time()
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                    
                content = await response.text()
                
                # Extract text content
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for element in soup(["script", "style"]):
                    element.decompose()
                    
                # Get text content
                text = soup.get_text(separator="\n", strip=True)
                
                # Extract metadata
                title = soup.title.string if soup.title else ""
                meta_desc = ""
                meta_desc_tag = soup.find("meta", attrs={"name": "description"})
                if meta_desc_tag:
                    meta_desc = meta_desc_tag.get("content", "")
                    
                # Create document
                doc = {
                    "url": url,
                    "title": title,
                    "description": meta_desc,
                    "content": text,
                    "timestamp": datetime.now().isoformat(),
                    "depth": depth
                }
                
                # Extract links for next depth
                if depth < self.max_depth:
                    links = await self.extract_links(url, content)
                    for link in links:
                        if len(self.crawled_urls) < self.max_pages:
                            await self.url_queue.put((link, depth + 1))
                            
                return doc
                
        except Exception as e:
            self.log_error(e, {"url": url})
            return None
            
    async def crawl(self, start_urls: List[str]) -> List[Dict[str, Any]]:
        """Crawl starting from given URLs.
        
        Args:
            start_urls: List of URLs to start crawling from
            
        Returns:
            List[Dict[str, Any]]: List of crawled documents
        """
        await self.initialize()
        
        # Initialize queue with start URLs
        for url in start_urls:
            await self.url_queue.put((url, 0))
            
        # Initialize workers
        workers = []
        results = []
        
        async def worker():
            while True:
                try:
                    # Get next URL from queue
                    url, depth = await self.url_queue.get()
                    
                    # Crawl URL
                    doc = await self.crawl_url(url, depth)
                    if doc:
                        results.append(doc)
                        
                    # Mark task as done
                    self.url_queue.task_done()
                    
                except asyncio.CancelledError:
                    break
                    
        # Start workers
        num_workers = 5  # Adjust based on needs
        for _ in range(num_workers):
            task = asyncio.create_task(worker())
            workers.append(task)
            
        # Wait for queue to be empty
        await self.url_queue.join()
        
        # Cancel workers
        for task in workers:
            task.cancel()
            
        await asyncio.gather(*workers, return_exceptions=True)
        await self.cleanup()
        
        return results
        
    def store_results(self, results: List[Dict[str, Any]]) -> None:
        """Store crawled results in database.
        
        Args:
            results: List of crawled documents
        """
        # Convert to LangChain documents
        documents = []
        for result in results:
            # Create content string
            content = f"Title: {result['title']}\n"
            content += f"Description: {result['description']}\n\n"
            content += result['content']
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "url": result["url"],
                    "timestamp": result["timestamp"],
                    "depth": result["depth"]
                }
            )
            documents.append(doc)
            
        # Split into chunks
        if documents:
            chunks = self.text_splitter.split_documents(documents)
            self.db.ingest_documents(chunks)
            
        # Log storage
        self.log_event(
            "results_stored",
            {
                "num_documents": len(documents),
                "num_chunks": len(chunks) if documents else 0
            }
        )
        
    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search crawled documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List[Document]: Matching documents
        """
        return self.db.search(query, k=k)
        
    def get_domain_stats(self) -> Dict[str, Any]:
        """Get crawling statistics per domain.
        
        Returns:
            Dict[str, Any]: Domain statistics
        """
        stats = {}
        for url in self.crawled_urls:
            domain = urlparse(url).netloc
            if domain not in stats:
                stats[domain] = {
                    "pages_crawled": 0,
                    "last_crawl": None
                }
            stats[domain]["pages_crawled"] += 1
            
        # Add last crawl times
        for domain, last_time in self.last_request.items():
            if domain in stats:
                stats[domain]["last_crawl"] = datetime.fromtimestamp(last_time)
                
        return stats