from typing import List, Union, Optional
from datetime import datetime
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llm_models.specialized_llm import SpecializedLLM
from datastores.worker_agent_db import WorkerAgentDB

class WorkerAgent:
    def __init__(self, name: str, domain_docs: Optional[Union[List[str], List[Document]]] = None):
        """Initialize a worker agent with a specific domain expertise.
        
        Args:
            name: Name/domain of the agent (e.g., "finance", "tech")
            domain_docs: Optional list of documents to initialize the knowledge base
        """
        self.name = name
        self.db = WorkerAgentDB(name)
        self.llm = SpecializedLLM(domain=self.name)
        
        # Create a custom prompt template for the QA chain
        qa_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question"]
        )
        
        # Initialize the QA chain with the custom prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm.get_llm(),
            chain_type="stuff",
            retriever=self.db.get_retriever(),
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        # Ingest initial documents if provided
        if domain_docs:
            self.ingest_knowledge(domain_docs)
            
    def ingest_knowledge(self, documents: Union[List[str], List[Document]], metadata: dict = None) -> None:
        """Add new documents to the agent's knowledge base.
        
        Args:
            documents: List of strings or Document objects to add
            metadata: Optional metadata to attach to the documents
        """
        self.db.ingest_documents(documents, metadata)

    def execute_task(self, task_description: str, context: Optional[str] = None) -> str:
        """Execute a task using the agent's knowledge and capabilities.
        
        Args:
            task_description: Description of the task to execute
            context: Optional additional context for the task
            
        Returns:
            str: The agent's response to the task
        """
        # If context is provided, use the specialized LLM's chain directly
        if context:
            return self.llm.generate_response(task_description, context)
            
        # Otherwise use the QA chain with the knowledge base
        return self.qa_chain.run(task_description)

    def communicate_with_other_agent(self, other_agent: 'WorkerAgent', query: str) -> dict:
        """Communicate with another agent to get additional information.
        
        Args:
            other_agent: Another WorkerAgent instance to communicate with
            query: The query to send to the other agent
            
        Returns:
            dict: Response containing the other agent's answer and metadata
        """
        # Get response from other agent
        other_response = other_agent.execute_task(query)
        
        # Return structured response with metadata
        return {
            "queried_agent": other_agent.name,
            "query": query,
            "response": other_response,
            "timestamp": datetime.now().isoformat()
        }
        
    def search_knowledge_base(self, query: str, k: int = 4) -> List[Document]:
        """Search the agent's knowledge base directly.
        
        Args:
            query: The search query
            k: Number of documents to return
            
        Returns:
            List[Document]: The most relevant documents
        """
        return self.db.search(query, k=k)
        
    def clear_knowledge(self) -> None:
        """Clear all documents from the agent's knowledge base."""
        self.db.clear_knowledge_base()
