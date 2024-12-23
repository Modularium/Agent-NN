from typing import List, Union, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datastores.vector_store import VectorStore

class WorkerAgentDB:
    def __init__(self, agent_name: str):
        """Initialize a database for a worker agent.
        
        Args:
            agent_name: Name of the agent, used to create a unique collection
        """
        self.agent_name = agent_name
        self.store = VectorStore(collection_name=f"{agent_name}_knowledge")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def ingest_documents(self, docs: Union[List[str], List[Document]], metadata: Dict[str, Any] = None) -> None:
        """Ingest documents into the vector store.
        
        Args:
            docs: List of strings or Document objects
            metadata: Optional metadata to attach to all documents
        """
        if not docs:
            return
            
        # Convert strings to Documents if necessary
        if isinstance(docs[0], str):
            documents = [Document(page_content=text, metadata=metadata or {}) for text in docs]
        else:
            documents = docs
            
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.store.add_documents(split_docs)

    def get_retriever(self, search_type: str = "similarity", k: int = 4):
        """Get a retriever interface to query the knowledge base.
        
        Args:
            search_type: Type of search to perform ("similarity" or "mmr")
            k: Number of documents to return
            
        Returns:
            A retriever object that can be used to query the vector store
        """
        return self.store.as_retriever(search_type=search_type, k=k)
    
    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search the knowledge base directly.
        
        Args:
            query: The search query
            k: Number of documents to return
            
        Returns:
            List of Document objects most similar to the query
        """
        return self.store.similarity_search(query, k=k)
    
    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base."""
        self.store.delete_collection()
