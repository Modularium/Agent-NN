from typing import List, Dict, Any, Optional, Union
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from agents.web_scraper_agent import WebScraperAgent
from agents.web_crawler_agent import WebCrawlerAgent
from datastores.vector_store import VectorStore
from utils.logging_util import LoggerMixin
from config.llm_config import OPENAI_CONFIG

class URLRAGSystem(LoggerMixin):
    """System for managing web-based RAG with automatic updates."""
    
    def __init__(self,
                 name: str = "url_rag",
                 update_interval: timedelta = timedelta(days=1),
                 embeddings_batch_size: int = 100):
        """Initialize URL-RAG system.
        
        Args:
            name: System name
            update_interval: How often to update content
            embeddings_batch_size: Batch size for embedding generation
        """
        super().__init__()
        self.name = name
        self.update_interval = update_interval
        self.embeddings_batch_size = embeddings_batch_size
        
        # Initialize agents
        self.scraper = WebScraperAgent()
        self.crawler = WebCrawlerAgent()
        
        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name=f"{name}_vectors",
            embedding_function=OpenAIEmbeddings(
                openai_api_key=OPENAI_CONFIG["api_key"]
            )
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Track processed URLs
        self.processed_urls: Dict[str, datetime] = {}
        
    async def scrape_and_index(self,
                              urls: List[str],
                              selectors: Dict[str, str],
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Scrape specific data and add to knowledge base.
        
        Args:
            urls: URLs to scrape
            selectors: CSS selectors for data extraction
            metadata: Optional metadata to store
        """
        # Scrape URLs
        results = await self.scraper.scrape_multiple_urls(
            urls,
            selectors,
            metadata
        )
        
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
                    "source": "scraper",
                    "url": result["source_url"],
                    "timestamp": result["timestamp"],
                    **(metadata or {})
                }
            )
            documents.append(doc)
            
            # Update processed URLs
            self.processed_urls[result["source_url"]] = datetime.now()
            
        # Add to vector store
        if documents:
            await self._add_to_vector_store(documents)
            
        # Log event
        self.log_event(
            "scraping_complete",
            {
                "num_urls": len(urls),
                "num_results": len(results),
                "num_documents": len(documents)
            }
        )
        
    async def crawl_and_index(self,
                             start_urls: List[str],
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Crawl websites and add to knowledge base.
        
        Args:
            start_urls: URLs to start crawling from
            metadata: Optional metadata to store
        """
        # Crawl URLs
        results = await self.crawler.crawl(start_urls)
        
        # Convert to documents
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
                    "source": "crawler",
                    "url": result["url"],
                    "timestamp": result["timestamp"],
                    "depth": result["depth"],
                    **(metadata or {})
                }
            )
            documents.append(doc)
            
            # Update processed URLs
            self.processed_urls[result["url"]] = datetime.now()
            
        # Add to vector store
        if documents:
            await self._add_to_vector_store(documents)
            
        # Log event
        self.log_event(
            "crawling_complete",
            {
                "num_start_urls": len(start_urls),
                "num_results": len(results),
                "num_documents": len(documents)
            }
        )
        
    async def _add_to_vector_store(self, documents: List[Document]) -> None:
        """Add documents to vector store in batches.
        
        Args:
            documents: Documents to add
        """
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Process in batches
        for i in range(0, len(chunks), self.embeddings_batch_size):
            batch = chunks[i:i + self.embeddings_batch_size]
            self.vector_store.add_documents(batch)
            
            # Log progress
            self.log_event(
                "batch_processed",
                {
                    "batch_size": len(batch),
                    "total_processed": i + len(batch),
                    "total_documents": len(chunks)
                }
            )
            
    async def update_content(self,
                           force: bool = False) -> None:
        """Update content that needs refreshing.
        
        Args:
            force: Force update all content
        """
        now = datetime.now()
        urls_to_update = []
        
        for url, last_update in self.processed_urls.items():
            if force or (now - last_update) > self.update_interval:
                urls_to_update.append(url)
                
        if not urls_to_update:
            return
            
        # Group URLs by source
        scraper_urls = []
        crawler_urls = []
        
        for url in urls_to_update:
            # Check source from existing documents
            docs = self.vector_store.similarity_search(
                f"url: {url}",
                k=1,
                filter={"url": url}
            )
            
            if docs and docs[0].metadata.get("source") == "scraper":
                scraper_urls.append(url)
            else:
                crawler_urls.append(url)
                
        # Update scraper content
        if scraper_urls:
            await self.scrape_and_index(
                scraper_urls,
                {},  # Need to store selectors somewhere
                {"update": True}
            )
            
        # Update crawler content
        if crawler_urls:
            await self.crawl_and_index(
                crawler_urls,
                {"update": True}
            )
            
        # Log update
        self.log_event(
            "content_updated",
            {
                "num_urls": len(urls_to_update),
                "num_scraper": len(scraper_urls),
                "num_crawler": len(crawler_urls)
            }
        )
        
    def search(self,
              query: str,
              k: int = 4,
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search knowledge base.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter criteria
            
        Returns:
            List[Document]: Matching documents
        """
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
        
    def get_source_stats(self) -> pd.DataFrame:
        """Get statistics about content sources.
        
        Returns:
            pd.DataFrame: Source statistics
        """
        # Get all documents
        docs = self.vector_store.similarity_search("", k=10000)
        
        # Create DataFrame
        data = []
        for doc in docs:
            data.append({
                "source": doc.metadata.get("source", "unknown"),
                "url": doc.metadata.get("url", ""),
                "timestamp": doc.metadata.get("timestamp", ""),
                "depth": doc.metadata.get("depth", 0)
            })
            
        df = pd.DataFrame(data)
        
        # Add age
        df["age_days"] = pd.to_datetime(df["timestamp"]).apply(
            lambda x: (datetime.now() - x).days
        )
        
        return df.groupby("source").agg({
            "url": "count",
            "age_days": ["mean", "min", "max"],
            "depth": ["mean", "max"]
        }).round(2)
        
    async def cleanup_old_content(self,
                                max_age: timedelta = timedelta(days=30)) -> None:
        """Remove old content from knowledge base.
        
        Args:
            max_age: Maximum age of content to keep
        """
        cutoff = datetime.now() - max_age
        
        # Get all documents
        docs = self.vector_store.similarity_search("", k=10000)
        
        # Find old documents
        old_docs = []
        for doc in docs:
            timestamp = doc.metadata.get("timestamp")
            if timestamp:
                doc_time = datetime.fromisoformat(timestamp)
                if doc_time < cutoff:
                    old_docs.append(doc)
                    
        # Remove old documents
        if old_docs:
            self.vector_store.delete_documents(
                [doc.metadata["url"] for doc in old_docs]
            )
            
            # Update processed URLs
            for doc in old_docs:
                self.processed_urls.pop(doc.metadata["url"], None)
                
        # Log cleanup
        self.log_event(
            "content_cleaned",
            {
                "max_age_days": max_age.days,
                "num_removed": len(old_docs)
            }
        )