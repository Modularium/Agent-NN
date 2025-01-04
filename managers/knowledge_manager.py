from typing import Dict, Any, Optional, List, Union, BinaryIO
import torch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredURLLoader,
    PyPDFLoader
)
import os
import json
import hashlib
import asyncio
import aiohttp
from datetime import datetime, timedelta
import mlflow
from utils.logging_util import LoggerMixin

class DocumentType:
    """Document types."""
    TEXT = "text"
    PDF = "pdf"
    CSV = "csv"
    URL = "url"

class KnowledgeManager(LoggerMixin):
    """Manager for knowledge base operations."""
    
    def __init__(self,
                 data_dir: str = "knowledge_bases",
                 embedding_model: str = "text-embedding-ada-002"):
        """Initialize manager.
        
        Args:
            data_dir: Data directory
            embedding_model: Embedding model name
        """
        super().__init__()
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize embeddings
        if os.getenv("OPENAI_API_KEY"):
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model
            )
        else:
            # Use mock embeddings for testing
            from langchain_core.embeddings import Embeddings
            class MockEmbeddings(Embeddings):
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.0] * 1536 for _ in texts]
                    
                def embed_query(self, text: str) -> List[float]:
                    return [0.0] * 1536
                    
            self.embeddings = MockEmbeddings()
        
        # Initialize MLflow
        self.experiment = mlflow.set_experiment("knowledge_management")
        
        # Initialize knowledge bases
        self.knowledge_bases: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
        
        # Initialize document processors
        self.processors = {
            DocumentType.TEXT: TextLoader,
            DocumentType.PDF: PyPDFLoader,
            DocumentType.CSV: CSVLoader,
            DocumentType.URL: UnstructuredURLLoader
        }
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def _load_registry(self):
        """Load knowledge base registry."""
        registry_path = os.path.join(self.data_dir, "registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                self.knowledge_bases = json.load(f)
                
    def _save_registry(self):
        """Save knowledge base registry."""
        registry_path = os.path.join(self.data_dir, "registry.json")
        
        # Convert knowledge bases to JSON-serializable format
        registry = {}
        for name, kb_info in self.knowledge_bases.items():
            registry[name] = {
                "name": kb_info["name"],
                "path": kb_info["path"],
                "description": kb_info.get("description", ""),
                "created_at": kb_info.get("created_at", ""),
                "updated_at": kb_info.get("updated_at", ""),
                "num_documents": kb_info.get("num_documents", 0),
                "num_chunks": kb_info.get("num_chunks", 0),
                "total_tokens": kb_info.get("total_tokens", 0),
                "embedding_model": kb_info.get("embedding_model", ""),
                "chunk_size": kb_info.get("chunk_size", 1000),
                "chunk_overlap": kb_info.get("chunk_overlap", 200)
            }
            
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
            
    def _get_kb_path(self, name: str) -> str:
        """Get knowledge base path.
        
        Args:
            name: Knowledge base name
            
        Returns:
            str: Knowledge base path
        """
        return os.path.join(self.data_dir, name)
        
    async def create_knowledge_base(self,
                                  name: str,
                                  domain: str,
                                  sources: List[str],
                                  update_interval: int = 3600) -> Dict[str, Any]:
        """Create knowledge base.
        
        Args:
            name: Knowledge base name
            domain: Knowledge domain
            sources: Data sources
            update_interval: Update interval in seconds
            
        Returns:
            Dict[str, Any]: Knowledge base information
        """
        if name in self.knowledge_bases:
            raise ValueError(f"Knowledge base exists: {name}")
            
        try:
            # Create directory
            kb_path = self._get_kb_path(name)
            os.makedirs(kb_path, exist_ok=True)
            
            # Initialize vector store
            vector_store = Chroma(
                collection_name=name,
                embedding_function=self.embeddings,
                persist_directory=kb_path
            )
            
            # Create knowledge base
            kb_info = {
                "name": name,
                "domain": domain,
                "sources": sources,
                "update_interval": update_interval,
                "created_at": datetime.now().isoformat(),
                "last_updated": None,
                "document_count": 0,
                "vector_store": vector_store,
                "path": kb_path
            }
            
            # Add to registry
            self.knowledge_bases[name] = kb_info
            self._save_registry()
            
            # Start initial ingestion
            await self._ingest_sources(name, sources)
            
            # Log creation
            self.log_event(
                "kb_created",
                {
                    "name": name,
                    "domain": domain,
                    "sources": sources
                }
            )
            
            return kb_info
            
        except Exception as e:
            self.log_error(e, {
                "name": name,
                "domain": domain
            })
            raise
            
    async def _ingest_sources(self,
                             kb_name: str,
                             sources: List[str]):
        """Ingest data sources.
        
        Args:
            kb_name: Knowledge base name
            sources: Data sources
        """
        kb_info = self.knowledge_bases[kb_name]
        vector_store = kb_info["vector_store"]
        
        for source in sources:
            try:
                if source.startswith("http"):
                    # Load URL
                    documents = await self._load_url(source)
                else:
                    # Load local file
                    documents = await self._load_file(source)
                    
                # Split documents
                texts = self.text_splitter.split_documents(documents)
                
                # Add to vector store
                vector_store.add_documents(texts)
                
                # Update count
                kb_info["document_count"] += len(texts)
                
            except Exception as e:
                self.log_error(e, {
                    "kb_name": kb_name,
                    "source": source
                })
                
        # Update timestamp
        kb_info["last_updated"] = datetime.now().isoformat()
        self._save_registry()
        
    async def _load_url(self, url: str) -> List[Any]:
        """Load URL content.
        
        Args:
            url: URL to load
            
        Returns:
            List[Any]: Loaded documents
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.text()
                
        # Create temporary file
        temp_path = os.path.join(
            self.data_dir,
            "temp",
            hashlib.md5(url.encode()).hexdigest()
        )
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, "w") as f:
            f.write(content)
            
        # Load content
        loader = self.processors[DocumentType.TEXT](temp_path)
        documents = loader.load()
        
        # Clean up
        os.remove(temp_path)
        
        return documents
        
    async def _load_file(self, path: str) -> List[Any]:
        """Load file content.
        
        Args:
            path: File path
            
        Returns:
            List[Any]: Loaded documents
        """
        # Determine file type
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            doc_type = DocumentType.TEXT
        elif ext == ".pdf":
            doc_type = DocumentType.PDF
        elif ext == ".csv":
            doc_type = DocumentType.CSV
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        # Load content
        loader = self.processors[doc_type](path)
        return loader.load()
        
    async def process_document(self,
                             kb_name: str,
                             filename: str,
                             content: bytes) -> str:
        """Process document.
        
        Args:
            kb_name: Knowledge base name
            filename: Document filename
            content: Document content
            
        Returns:
            str: Document ID
        """
        if kb_name not in self.knowledge_bases:
            raise ValueError(f"Unknown knowledge base: {kb_name}")
            
        try:
            # Create document ID
            doc_id = hashlib.md5(content).hexdigest()
            
            # Save content
            kb_path = self._get_kb_path(kb_name)
            doc_path = os.path.join(kb_path, "documents", doc_id + ".txt")
            os.makedirs(os.path.dirname(doc_path), exist_ok=True)
            
            with open(doc_path, "wb") as f:
                f.write(content)
                
            # Process document
            documents = await self._load_file(doc_path)
            texts = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            kb_info = self.knowledge_bases[kb_name]
            vector_store = kb_info["vector_store"]
            vector_store.add_documents(texts)
            
            # Update count
            kb_info["document_count"] += len(texts)
            kb_info["last_updated"] = datetime.now().isoformat()
            self._save_registry()
            
            # Log processing
            self.log_event(
                "document_processed",
                {
                    "kb_name": kb_name,
                    "filename": filename,
                    "doc_id": doc_id
                }
            )
            
            return doc_id
            
        except Exception as e:
            self.log_error(e, {
                "kb_name": kb_name,
                "filename": filename
            })
            raise
            
    def search_knowledge_base(self,
                            kb_name: str,
                            query: str,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge base.
        
        Args:
            kb_name: Knowledge base name
            query: Search query
            limit: Maximum results
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        if kb_name not in self.knowledge_bases:
            raise ValueError(f"Unknown knowledge base: {kb_name}")
            
        try:
            # Get vector store
            kb_info = self.knowledge_bases[kb_name]
            vector_store = kb_info["vector_store"]
            
            # Search documents
            results = vector_store.similarity_search_with_score(
                query,
                k=limit
            )
            
            # Format results
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            self.log_error(e, {
                "kb_name": kb_name,
                "query": query
            })
            raise
            
    def get_knowledge_base_info(self,
                              kb_name: str) -> Optional[Dict[str, Any]]:
        """Get knowledge base information.
        
        Args:
            kb_name: Knowledge base name
            
        Returns:
            Optional[Dict[str, Any]]: Knowledge base information
        """
        if kb_name not in self.knowledge_bases:
            return None
            
        kb_info = self.knowledge_bases[kb_name].copy()
        del kb_info["vector_store"]  # Remove vector store
        return kb_info
        
    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """List knowledge bases.
        
        Returns:
            List[Dict[str, Any]]: Knowledge base information
        """
        return [
            self.get_knowledge_base_info(name)
            for name in self.knowledge_bases.keys()
        ]
        
    async def update_knowledge_base(self, kb_name: str):
        """Update knowledge base.
        
        Args:
            kb_name: Knowledge base name
        """
        if kb_name not in self.knowledge_bases:
            raise ValueError(f"Unknown knowledge base: {kb_name}")
            
        try:
            kb_info = self.knowledge_bases[kb_name]
            
            # Check update interval
            if kb_info["last_updated"]:
                last_update = datetime.fromisoformat(
                    kb_info["last_updated"]
                )
                elapsed = datetime.now() - last_update
                if elapsed.total_seconds() < kb_info["update_interval"]:
                    return
                    
            # Update sources
            await self._ingest_sources(
                kb_name,
                kb_info["sources"]
            )
            
            # Log update
            self.log_event(
                "kb_updated",
                {"name": kb_name}
            )
            
        except Exception as e:
            self.log_error(e, {"kb_name": kb_name})
            raise
            
    async def delete_knowledge_base(self, kb_name: str):
        """Delete knowledge base.
        
        Args:
            kb_name: Knowledge base name
        """
        if kb_name not in self.knowledge_bases:
            return
            
        try:
            # Remove directory
            kb_path = self._get_kb_path(kb_name)
            if os.path.exists(kb_path):
                shutil.rmtree(kb_path)
                
            # Remove from registry
            del self.knowledge_bases[kb_name]
            self._save_registry()
            
            # Log deletion
            self.log_event(
                "kb_deleted",
                {"name": kb_name}
            )
            
        except Exception as e:
            self.log_error(e, {"kb_name": kb_name})
            raise