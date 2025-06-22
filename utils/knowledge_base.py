"""Utilities for managing domain-specific knowledge bases."""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.logging_util import setup_logger

logger = setup_logger(__name__)

class KnowledgeBaseManager:
    """Manager for domain-specific knowledge bases."""
    
    # Supported file types and their loaders
    SUPPORTED_LOADERS = {
        ".txt": TextLoader,
        ".pdf": PDFLoader,
        ".csv": CSVLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": UnstructuredHTMLLoader
    }
    
    def __init__(self, base_path: str):
        """Initialize the knowledge base manager.
        
        Args:
            base_path: Base path for storing knowledge base files
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        # Create domain-specific directories
        self.domain_paths = {
            domain: os.path.join(base_path, domain)
            for domain in ["finance", "tech", "marketing"]
        }
        for path in self.domain_paths.values():
            os.makedirs(path, exist_ok=True)
            
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Load document metadata
        self.metadata_path = os.path.join(base_path, "metadata.json")
        self.document_metadata = self._load_metadata()
        
    def ingest_document(self,
                       file_path: str,
                       domain: str,
                       metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Ingest a document into the knowledge base.
        
        Args:
            file_path: Path to the document
            domain: Domain this document belongs to
            metadata: Optional additional metadata
            
        Returns:
            List of Document objects created from the file
        """
        if domain not in self.domain_paths:
            raise ValueError(f"Unsupported domain: {domain}")
            
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
            target_path = os.path.join(
                self.domain_paths[domain],
                os.path.basename(file_path)
            )
            if not os.path.exists(target_path):
                with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())
            
            logger.info(f"Ingested document {file_path} into {domain} domain")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {str(e)}")
            raise
            
    def get_domain_documents(self,
                           domain: str,
                           file_type: Optional[str] = None,
                           after_date: Optional[str] = None) -> List[Document]:
        """Get all documents for a specific domain.
        
        Args:
            domain: Domain to get documents for
            file_type: Optional filter by file type
            after_date: Optional filter for documents after this date
            
        Returns:
            List of Document objects
        """
        if domain not in self.domain_paths:
            raise ValueError(f"Unsupported domain: {domain}")
            
        domain_path = self.domain_paths[domain]
        documents = []
        
        for file_name in os.listdir(domain_path):
            file_path = os.path.join(domain_path, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # Apply filters
            if file_type and file_ext != file_type:
                continue
                
            metadata = self.document_metadata.get(file_path, {})
            if after_date and metadata.get("ingestion_timestamp", "") < after_date:
                continue
                
            if file_ext in self.SUPPORTED_LOADERS:
                try:
                    loader = self.SUPPORTED_LOADERS[file_ext](file_path)
                    docs = loader.load()
                    split_docs = self.text_splitter.split_documents(docs)
                    
                    # Add metadata
                    for doc in split_docs:
                        doc.metadata.update(metadata)
                    
                    documents.extend(split_docs)
                except Exception as e:
                    logger.error(f"Error loading document {file_path}: {str(e)}")
                    
        return documents
        
    def search_documents(self,
                        query: str,
                        domain: Optional[str] = None,
                        limit: int = 5) -> List[Document]:
        """Search for documents matching a query.
        
        Args:
            query: Search query
            domain: Optional domain to search in
            limit: Maximum number of documents to return
            
        Returns:
            List of matching Document objects
        """
        # This is a placeholder for more sophisticated search
        # In practice, this would use the vector store's similarity search
        matching_docs = []
        domains = [domain] if domain else self.domain_paths.keys()
        
        for d in domains:
            docs = self.get_domain_documents(d)
            for doc in docs:
                if query.lower() in doc.page_content.lower():
                    matching_docs.append(doc)
                    if len(matching_docs) >= limit:
                        return matching_docs
                        
        return matching_docs
        
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a specific document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dict containing document metadata
        """
        return self.document_metadata.get(file_path, {})
        
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from disk."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                return {}
        return {}
        
    def _save_document_metadata(self, file_path: str, metadata: Dict[str, Any]):
        """Save document metadata to disk."""
        self.document_metadata[file_path] = metadata
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.document_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            
    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Get statistics about a domain's knowledge base.
        
        Args:
            domain: Domain to get statistics for
            
        Returns:
            Dict containing domain statistics
        """
        if domain not in self.domain_paths:
            raise ValueError(f"Unsupported domain: {domain}")
            
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