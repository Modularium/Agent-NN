from typing import Dict, Any, Optional, List, Union
import torch
from datetime import datetime
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datastores.vector_store import VectorStore
from utils.logging_util import LoggerMixin
from config.llm_config import OPENAI_CONFIG

class DomainKnowledgeManager(LoggerMixin):
    """Manager for domain-specific knowledge bases."""
    
    def __init__(self,
                 base_path: str = "knowledge_bases",
                 embedding_batch_size: int = 100):
        """Initialize domain knowledge manager.
        
        Args:
            base_path: Base path for knowledge bases
            embedding_batch_size: Batch size for embeddings
        """
        super().__init__()
        self.base_path = base_path
        self.embedding_batch_size = embedding_batch_size
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_CONFIG["api_key"]
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Domain vector stores
        self.vector_stores: Dict[str, VectorStore] = {}
        
        # Domain metadata
        self.domain_metadata: Dict[str, Dict[str, Any]] = {}
        
    def add_domain(self,
                  domain: str,
                  description: str,
                  initial_docs: Optional[List[Union[str, Document]]] = None):
        """Add new domain knowledge base.
        
        Args:
            domain: Domain name
            description: Domain description
            initial_docs: Optional initial documents
        """
        # Create vector store
        vector_store = VectorStore(
            collection_name=f"{domain}_knowledge",
            embedding_function=self.embeddings
        )
        self.vector_stores[domain] = vector_store
        
        # Store metadata
        self.domain_metadata[domain] = {
            "description": description,
            "created_at": datetime.now().isoformat(),
            "document_count": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Add initial documents
        if initial_docs:
            self.add_documents(domain, initial_docs)
            
        # Log domain creation
        self.log_event(
            "domain_created",
            {
                "domain": domain,
                "initial_docs": len(initial_docs) if initial_docs else 0
            }
        )
        
    def add_documents(self,
                     domain: str,
                     documents: List[Union[str, Document]],
                     metadata: Optional[Dict[str, Any]] = None):
        """Add documents to domain knowledge base.
        
        Args:
            domain: Domain name
            documents: Documents to add
            metadata: Optional metadata
        """
        if domain not in self.vector_stores:
            raise ValueError(f"Unknown domain: {domain}")
            
        # Convert strings to documents
        docs = []
        for doc in documents:
            if isinstance(doc, str):
                docs.append(Document(
                    page_content=doc,
                    metadata={
                        "domain": domain,
                        "added_at": datetime.now().isoformat(),
                        **(metadata or {})
                    }
                ))
            else:
                # Update document metadata
                doc.metadata.update({
                    "domain": domain,
                    "added_at": datetime.now().isoformat(),
                    **(metadata or {})
                })
                docs.append(doc)
                
        # Split documents
        chunks = self.text_splitter.split_documents(docs)
        
        # Add to vector store in batches
        for i in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[i:i + self.embedding_batch_size]
            self.vector_stores[domain].add_documents(batch)
            
        # Update metadata
        self.domain_metadata[domain]["document_count"] += len(docs)
        self.domain_metadata[domain]["last_updated"] = datetime.now().isoformat()
        
        # Log addition
        self.log_event(
            "documents_added",
            {
                "domain": domain,
                "count": len(docs),
                "total_chunks": len(chunks)
            }
        )
        
    def search_domain(self,
                     domain: str,
                     query: str,
                     k: int = 4) -> List[Document]:
        """Search domain knowledge base.
        
        Args:
            domain: Domain name
            query: Search query
            k: Number of results
            
        Returns:
            List[Document]: Matching documents
        """
        if domain not in self.vector_stores:
            raise ValueError(f"Unknown domain: {domain}")
            
        return self.vector_stores[domain].similarity_search(query, k=k)
        
    def search_all_domains(self,
                         query: str,
                         k: int = 4) -> Dict[str, List[Document]]:
        """Search all domain knowledge bases.
        
        Args:
            query: Search query
            k: Number of results per domain
            
        Returns:
            Dict[str, List[Document]]: Results by domain
        """
        results = {}
        for domain in self.vector_stores:
            results[domain] = self.search_domain(domain, query, k=k)
            
        return results
        
    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """Get domain information.
        
        Args:
            domain: Domain name
            
        Returns:
            Dict[str, Any]: Domain information
        """
        if domain not in self.domain_metadata:
            raise ValueError(f"Unknown domain: {domain}")
            
        info = self.domain_metadata[domain].copy()
        
        # Add vector store stats
        vector_store = self.vector_stores[domain]
        info["embedding_count"] = len(vector_store)
        
        return info
        
    def get_all_domains(self) -> List[str]:
        """Get list of all domains.
        
        Returns:
            List[str]: Domain names
        """
        return list(self.domain_metadata.keys())
        
    def remove_domain(self, domain: str):
        """Remove domain knowledge base.
        
        Args:
            domain: Domain name
        """
        if domain not in self.vector_stores:
            raise ValueError(f"Unknown domain: {domain}")
            
        # Remove vector store
        self.vector_stores[domain].clear()
        del self.vector_stores[domain]
        
        # Remove metadata
        del self.domain_metadata[domain]
        
        # Log removal
        self.log_event(
            "domain_removed",
            {"domain": domain}
        )
        
    def clear_domain(self, domain: str):
        """Clear all documents from domain.
        
        Args:
            domain: Domain name
        """
        if domain not in self.vector_stores:
            raise ValueError(f"Unknown domain: {domain}")
            
        # Clear vector store
        self.vector_stores[domain].clear()
        
        # Update metadata
        self.domain_metadata[domain]["document_count"] = 0
        self.domain_metadata[domain]["last_updated"] = datetime.now().isoformat()
        
        # Log clear
        self.log_event(
            "domain_cleared",
            {"domain": domain}
        )
        
    def get_domain_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all domains.
        
        Returns:
            Dict[str, Dict[str, Any]]: Domain statistics
        """
        stats = {}
        for domain in self.domain_metadata:
            info = self.get_domain_info(domain)
            stats[domain] = {
                "document_count": info["document_count"],
                "embedding_count": info["embedding_count"],
                "last_updated": info["last_updated"]
            }
            
        return stats