from typing import List, Union, Optional, Dict, Any
from datetime import datetime
import asyncio
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llm_models.specialized_llm import SpecializedLLM
from datastores.worker_agent_db import WorkerAgentDB
from .agent_communication import AgentCommunicationHub, AgentMessage, MessageType
from .domain_knowledge import DomainKnowledgeManager

class WorkerAgent:
    def __init__(self,
                 name: str,
                 domain_docs: Optional[Union[List[str], List[Document]]] = None,
                 communication_hub: Optional[AgentCommunicationHub] = None,
                 knowledge_manager: Optional[DomainKnowledgeManager] = None):
        """Initialize a worker agent with a specific domain expertise.
        
        Args:
            name: Name/domain of the agent (e.g., "finance", "tech")
            domain_docs: Optional list of documents to initialize the knowledge base
            communication_hub: Optional communication hub for agent interaction
            knowledge_manager: Optional domain knowledge manager
        """
        self.name = name
        self.db = WorkerAgentDB(name)
        self.llm = SpecializedLLM(domain=self.name)
        
        # Communication and knowledge management
        self.communication_hub = communication_hub
        self.knowledge_manager = knowledge_manager
        
        # Create a custom prompt template for the QA chain
        qa_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        If you need information from another domain, indicate that in your response.
        
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
            
        # Register with communication hub
        if self.communication_hub:
            asyncio.create_task(self.communication_hub.register_agent(self.name))
            
    async def process_messages(self):
        """Process incoming messages."""
        if not self.communication_hub:
            return
            
        while True:
            messages = await self.communication_hub.get_messages(self.name)
            for message in messages:
                await self._handle_message(message)
            await asyncio.sleep(0.1)  # Avoid busy waiting
            
    async def _handle_message(self, message: AgentMessage):
        """Handle an incoming message.
        
        Args:
            message: Message to handle
        """
        try:
            if message.message_type == MessageType.QUERY:
                # Handle information query
                response = self.execute_task(message.content)
                await self._send_response(message.sender, response)
                
            elif message.message_type == MessageType.CLARIFICATION:
                # Handle clarification request
                context = message.metadata.get("context", {})
                response = self.execute_task(message.content, str(context))
                await self._send_response(
                    message.sender,
                    response,
                    MessageType.CLARIFICATION
                )
                
            elif message.message_type == MessageType.TASK:
                # Handle task delegation
                requirements = message.metadata.get("requirements", {})
                result = self.execute_task(
                    message.content,
                    str(requirements)
                )
                await self._send_response(
                    message.sender,
                    result,
                    MessageType.RESULT
                )
                
            elif message.message_type == MessageType.UPDATE:
                # Handle knowledge update
                if "knowledge" in message.metadata:
                    self.ingest_knowledge(
                        [message.metadata["knowledge"]],
                        {"source": f"update_from_{message.sender}"}
                    )
                    
        except Exception as e:
            # Send error message
            await self._send_response(
                message.sender,
                str(e),
                MessageType.ERROR
            )
            
    async def _send_response(self,
                           receiver: str,
                           content: str,
                           message_type: MessageType = MessageType.RESPONSE):
        """Send a response message.
        
        Args:
            receiver: Message recipient
            content: Message content
            message_type: Type of message
        """
        if self.communication_hub:
            message = AgentMessage(
                message_type=message_type,
                sender=self.name,
                receiver=receiver,
                content=content,
                metadata={}
            )
            await self.communication_hub.send_message(message)
            
    def ingest_knowledge(self,
                        documents: Union[List[str], List[Document]],
                        metadata: dict = None) -> None:
        """Add new documents to the agent's knowledge base.
        
        Args:
            documents: List of strings or Document objects to add
            metadata: Optional metadata to attach to the documents
        """
        self.db.ingest_documents(documents, metadata)
        
        # Add to domain knowledge if available
        if self.knowledge_manager:
            for doc in documents:
                content = doc.page_content if isinstance(doc, Document) else doc
                self.knowledge_manager.add_knowledge(
                    domain=self.name,
                    content=content,
                    metadata=metadata
                )

    async def execute_task(self,
                          task_description: str,
                          context: Optional[str] = None) -> str:
        """Execute a task using the agent's knowledge and capabilities.
        
        Args:
            task_description: Description of the task to execute
            context: Optional additional context for the task
            
        Returns:
            str: The agent's response to the task
        """
        try:
            # Check if we need information from other domains
            if self.knowledge_manager:
                related_knowledge = self.knowledge_manager.search_knowledge(
                    task_description
                )
                if related_knowledge:
                    other_domains = set(
                        node.domain for node in related_knowledge
                        if node.domain != self.name
                    )
                    if other_domains and self.communication_hub:
                        # Get information from other domains
                        responses = await self._gather_domain_knowledge(
                            task_description,
                            other_domains
                        )
                        if responses:
                            context = (context or "") + "\n" + "\n".join(responses)
                            
            # Execute task
            if context:
                return self.llm.generate_response(task_description, context)
            else:
                return self.qa_chain.run(task_description)
                
        except Exception as e:
            # Log error and return error message
            logger.error(f"Error executing task: {str(e)}")
            return f"Error executing task: {str(e)}"
            
    async def _gather_domain_knowledge(self,
                                     query: str,
                                     domains: set) -> List[str]:
        """Gather knowledge from other domains.
        
        Args:
            query: Query to send
            domains: Domains to query
            
        Returns:
            List of responses
        """
        responses = []
        for domain in domains:
            message = AgentMessage(
                message_type=MessageType.QUERY,
                sender=self.name,
                receiver=f"{domain}_agent",
                content=query,
                metadata={"purpose": "domain_knowledge"}
            )
            await self.communication_hub.send_message(message)
            
            # Wait for response (with timeout)
            try:
                async with asyncio.timeout(5):
                    while True:
                        messages = await self.communication_hub.get_messages(
                            self.name
                        )
                        for msg in messages:
                            if (msg.sender == f"{domain}_agent" and
                                msg.message_type == MessageType.RESPONSE):
                                responses.append(
                                    f"From {domain}: {msg.content}"
                                )
                                break
                        if len(responses) == len(domains):
                            break
                        await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for response from {domain}")
                
        return responses
        
    async def communicate_with_other_agent(self,
                                         other_agent: str,
                                         query: str,
                                         timeout: float = 5.0) -> Dict[str, Any]:
        """Communicate with another agent to get additional information.
        
        Args:
            other_agent: Name of agent to communicate with
            query: Query to send
            timeout: Timeout in seconds
            
        Returns:
            Dict containing response and metadata
        """
        if not self.communication_hub:
            raise ValueError("Communication hub not available")
            
        # Send query
        message = AgentMessage(
            message_type=MessageType.QUERY,
            sender=self.name,
            receiver=other_agent,
            content=query,
            metadata={}
        )
        await self.communication_hub.send_message(message)
        
        # Wait for response
        try:
            async with asyncio.timeout(timeout):
                while True:
                    messages = await self.communication_hub.get_messages(self.name)
                    for msg in messages:
                        if (msg.sender == other_agent and
                            msg.message_type == MessageType.RESPONSE):
                            return {
                                "queried_agent": other_agent,
                                "query": query,
                                "response": msg.content,
                                "timestamp": msg.timestamp
                            }
                    await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            return {
                "queried_agent": other_agent,
                "query": query,
                "error": "Timeout waiting for response",
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
        
    async def shutdown(self):
        """Clean up resources."""
        if self.communication_hub:
            await self.communication_hub.deregister_agent(self.name)
