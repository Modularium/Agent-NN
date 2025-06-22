"""Manager for domain-specific documents."""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
from langchain.schema import Document
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.logging_util import setup_logger

logger = setup_logger(__name__)

class DocumentManager:
    """Manager for domain-specific documents."""
    
    # Supported file types and their loaders
    SUPPORTED_LOADERS = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".csv": CSVLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": UnstructuredHTMLLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".docx": UnstructuredWordLoader
    }
    
    # Domain-specific document sources
    DOMAIN_SOURCES = {
        "finance": [
            "https://www.sec.gov/edgar/searchedgar/companysearch",
            "https://www.federalreserve.gov/publications.htm",
            "https://www.investopedia.com/financial-term-dictionary-4769738"
        ],
        "tech": [
            "https://docs.python.org/3/",
            "https://docs.docker.com/",
            "https://kubernetes.io/docs/home/"
        ],
        "marketing": [
            "https://blog.hubspot.com/marketing",
            "https://www.marketingweek.com/",
            "https://www.thinkwithgoogle.com/"
        ],
        "legal": [
            "https://www.law.cornell.edu/",
            "https://www.findlaw.com/",
            "https://www.lexisnexis.com/"
        ]
    }
    
    def __init__(self, base_dir: str = "data/documents"):
        """Initialize document manager.
        
        Args:
            base_dir: Base directory for documents
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create domain directories
        self.domain_dirs = {}
        for domain in self.DOMAIN_SOURCES:
            domain_dir = self.base_dir / domain
            domain_dir.mkdir(exist_ok=True)
            self.domain_dirs[domain] = domain_dir
            
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Document metadata
        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def ingest_file(self,
                    file_path: str,
                    domain: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest a file into the document store.
        
        Args:
            file_path: Path to file
            domain: Document domain
            metadata: Optional metadata
            
        Returns:
            List of created documents
        """
        if domain not in self.domain_dirs:
            raise ValueError(f"Unknown domain: {domain}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.SUPPORTED_LOADERS:
            raise ValueError(f"Unsupported file type: {file_ext}")
            
        try:
            # Load document
            loader = self.SUPPORTED_LOADERS[file_ext](file_path)
            documents = loader.load()
            
            # Split into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add metadata
            doc_metadata = {
                "domain": domain,
                "source_file": os.path.basename(file_path),
                "ingestion_timestamp": datetime.now().isoformat(),
                "file_type": file_ext,
                **(metadata or {})
            }
            
            for doc in split_docs:
                doc.metadata.update(doc_metadata)
                
            # Save document metadata
            self._save_document_metadata(file_path, doc_metadata)
            
            # Copy file to domain directory
            target_path = self.domain_dirs[domain] / os.path.basename(file_path)
            if not target_path.exists():
                with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())
                    
            logger.info(f"Ingested document {file_path} into {domain} domain")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}")
            raise
            
    def ingest_url(self,
                   url: str,
                   domain: str,
                   metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest content from a URL.
        
        Args:
            url: URL to ingest
            domain: Document domain
            metadata: Optional metadata
            
        Returns:
            List of created documents
        """
        try:
            # Download content
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (customize based on site structure)
            content = soup.get_text()
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "domain": domain,
                    "source_url": url,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "file_type": "url",
                    **(metadata or {})
                }
            )
            
            # Split into chunks
            split_docs = self.text_splitter.split_documents([doc])
            
            # Save metadata
            self._save_document_metadata(url, doc.metadata)
            
            logger.info(f"Ingested URL {url} into {domain} domain")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error ingesting URL {url}: {str(e)}")
            raise
            
    def ingest_text(self,
                    text: str,
                    domain: str,
                    metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest raw text.
        
        Args:
            text: Text content
            domain: Document domain
            metadata: Optional metadata
            
        Returns:
            List of created documents
        """
        try:
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "domain": domain,
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "file_type": "text",
                    **(metadata or {})
                }
            )
            
            # Split into chunks
            split_docs = self.text_splitter.split_documents([doc])
            
            # Save metadata
            doc_id = f"text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_document_metadata(doc_id, doc.metadata)
            
            logger.info(f"Ingested text into {domain} domain")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error ingesting text: {str(e)}")
            raise
            
    def get_domain_documents(self,
                           domain: str,
                           file_type: Optional[str] = None) -> List[Document]:
        """Get all documents for a domain.
        
        Args:
            domain: Domain to get documents for
            file_type: Optional file type filter
            
        Returns:
            List of documents
        """
        if domain not in self.domain_dirs:
            raise ValueError(f"Unknown domain: {domain}")
            
        documents = []
        domain_dir = self.domain_dirs[domain]
        
        for file_path in domain_dir.glob("*"):
            file_ext = file_path.suffix.lower()
            
            # Apply file type filter
            if file_type and file_ext != file_type:
                continue
                
            if file_ext in self.SUPPORTED_LOADERS:
                try:
                    loader = self.SUPPORTED_LOADERS[file_ext](str(file_path))
                    docs = loader.load()
                    
                    # Add metadata
                    metadata = self.metadata.get(str(file_path), {})
                    for doc in docs:
                        doc.metadata.update(metadata)
                        
                    documents.extend(docs)
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
        return documents
        
    def search_documents(self,
                        query: str,
                        domain: Optional[str] = None,
                        limit: int = 5) -> List[Document]:
        """Search for documents.
        
        Args:
            query: Search query
            domain: Optional domain filter
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        # This is a simple text search
        # In practice, you'd want to use embeddings and vector search
        matching_docs = []
        domains = [domain] if domain else self.domain_dirs.keys()
        
        for d in domains:
            docs = self.get_domain_documents(d)
            for doc in docs:
                if query.lower() in doc.page_content.lower():
                    matching_docs.append(doc)
                    if len(matching_docs) >= limit:
                        return matching_docs
                        
        return matching_docs
        
    def get_document_metadata(self, identifier: str) -> Dict[str, Any]:
        """Get metadata for a document.
        
        Args:
            identifier: Document identifier (path or URL)
            
        Returns:
            Dict containing metadata
        """
        return self.metadata.get(identifier, {})
        
    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dict containing statistics
        """
        if domain not in self.domain_dirs:
            raise ValueError(f"Unknown domain: {domain}")
            
        docs = self.get_domain_documents(domain)
        
        return {
            "total_documents": len(docs),
            "total_tokens": sum(len(doc.page_content.split()) for doc in docs),
            "file_types": {
                ext: sum(1 for doc in docs if doc.metadata.get("file_type") == ext)
                for ext in self.SUPPORTED_LOADERS.keys()
            },
            "latest_update": max(
                (doc.metadata.get("ingestion_timestamp", "")
                 for doc in docs),
                default=None
            )
        }
        
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                return {}
        return {}
        
    def _save_document_metadata(self, identifier: str, metadata: Dict[str, Any]):
        """Save document metadata to disk.
        
        Args:
            identifier: Document identifier
            metadata: Document metadata
        """
        self.metadata[identifier] = metadata
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            
    def export_domain_data(self,
                          domain: str,
                          output_format: str = "csv") -> str:
        """Export domain data to a file.
        
        Args:
            domain: Domain to export
            output_format: Output format (csv or json)
            
        Returns:
            str: Path to exported file
        """
        if domain not in self.domain_dirs:
            raise ValueError(f"Unknown domain: {domain}")
            
        docs = self.get_domain_documents(domain)
        
        # Convert to list of dicts
        data = []
        for doc in docs:
            data.append({
                "content": doc.page_content,
                **doc.metadata
            })
            
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.base_dir / f"{domain}_export_{timestamp}.{output_format}"
        
        try:
            if output_format == "csv":
                df = pd.DataFrame(data)
                df.to_csv(output_file, index=False)
            else:
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            logger.info(f"Exported {domain} data to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise