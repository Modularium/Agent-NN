import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from config.llm_config import OPENAI_CONFIG
from config import VECTOR_DB_PATH

LLM_API_KEY = OPENAI_CONFIG["api_key"]

class VectorStore:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
        # Use OpenAI embeddings if API key is available, otherwise use HuggingFace
        if LLM_API_KEY:
            self.embedding_function = OpenAIEmbeddings(openai_api_key=LLM_API_KEY)
        else:
            self.embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
        

        # Initialize Chroma
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, docs: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            docs: List of Document objects with page_content and metadata
        """
        self.vector_store.add_documents(docs)

    def as_retriever(self, search_type: str = "similarity", k: int = 4):
        """Get a retriever interface to the vector store.
        
        Args:
            search_type: Type of search to perform ("similarity" or "mmr")
            k: Number of documents to return
            
        Returns:
            A retriever object that can be used to query the vector store
        """
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform a similarity search directly.
        
        Args:
            query: The search query
            k: Number of documents to return
            
        Returns:
            List of Document objects most similar to the query
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def delete_collection(self) -> None:
        """Delete the entire collection from the vector store."""
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=VECTOR_DB_PATH
        )
