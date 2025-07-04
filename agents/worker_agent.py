from typing import List, Union, Optional, Dict, Any
from datetime import datetime
import asyncio
import time
import torch
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llm_models.specialized_llm import SpecializedLLM
from datastores.worker_agent_db import WorkerAgentDB
from .agent_communication import AgentCommunicationHub, AgentMessage, MessageType
from .domain_knowledge import DomainKnowledgeManager
from nn_models.agent_nn_v2 import AgentNN, TaskMetrics
from utils.logging_util import LoggerMixin

class WorkerAgent(LoggerMixin):
    def __init__(
        self,
        name: str,
        domain_docs: Optional[Union[List[str], List[Document]]] = None,
        communication_hub: Optional[AgentCommunicationHub] = None,
        knowledge_manager: Optional[DomainKnowledgeManager] = None,
        *,
        use_nn_features: bool = True,
    ):
        """Initialize a worker agent with a specific domain expertise.
        
        Args:
            name: Name/domain of the agent (e.g., "finance", "tech")
            domain_docs: Optional list of documents to initialize the knowledge base
            communication_hub: Optional communication hub for agent interaction
            knowledge_manager: Optional domain knowledge manager
            use_nn_features: Enable neural network based task features
        """
        super().__init__()
        self.name = name
        self.db = WorkerAgentDB(name)
        self.llm = SpecializedLLM(domain=self.name)
        
        # Initialize neural network
        self.use_nn_features = use_nn_features
        self.nn = AgentNN(domain=name)
        if self.use_nn_features:
            self.load_or_init_nn()
        
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
            start_time = time.time()
            task_features_str = ""
            if self.use_nn_features:
                task_embedding = self.get_task_embedding(task_description)
                task_features = self.nn.predict_task_features(task_embedding)
                task_features_str = self.format_task_features(task_features)

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
            if self.use_nn_features and hasattr(self.llm, "generate_with_confidence"):
                if context:
                    response, confidence = self.llm.generate_with_confidence(
                        task_description,
                        context=context,
                        task_features=task_features_str,
                    )
                else:
                    response, confidence = self.llm.generate_with_confidence(
                        task_description,
                        task_features=task_features_str,
                    )
            else:
                if context:
                    response = self.llm.generate_response(task_description, context)
                else:
                    response = self.qa_chain.run(task_description)
                confidence = 0.0

            if self.use_nn_features:
                metrics = TaskMetrics(
                    response_time=time.time() - start_time,
                    confidence_score=confidence,
                )
                self.nn.evaluate_performance(metrics)

            return response
                
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

    def get_task_embedding(self, task_description: str) -> torch.Tensor:
        """Return embedding tensor for a task description."""
        embedding = self.llm.get_embedding(task_description)
        return torch.tensor(embedding).unsqueeze(0)

    def format_task_features(self, features: torch.Tensor) -> str:
        """Format feature tensor into a readable string."""
        values = features.squeeze().tolist()
        return "\n".join(
            [f"Feature {i + 1}: {val:.3f}" for i, val in enumerate(values)]
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Return aggregated neural network training metrics."""
        summary = self.nn.get_training_summary()
        recent = self.nn.eval_metrics[-100:]
        if recent:
            avg_metrics = {
                "avg_response_time": sum(m["response_time"] for m in recent) / len(recent),
                "avg_confidence": sum(m["confidence"] for m in recent) / len(recent),
            }
            feedback = [m["user_feedback"] for m in recent if "user_feedback" in m]
            if feedback:
                avg_metrics["avg_user_feedback"] = sum(feedback) / len(feedback)
            success = [m["success_rate"] for m in recent if "success_rate" in m]
            if success:
                avg_metrics["success_rate"] = sum(success) / len(success)
            summary.update(avg_metrics)
        return summary
        
    def get_features(self) -> torch.Tensor:
        """Get neural network features for the agent.
        
        Returns:
            torch.Tensor: Feature tensor
        """
        # Get agent description
        description = f"Domain: {self.name}\n"
        description += "\n".join(
            doc.page_content
            for doc in self.search_knowledge_base("", k=5)
        )
        
        # Get embedding from LLM
        embedding = torch.tensor(
            self.llm.get_embedding(description)
        ).unsqueeze(0)
        
        # Get features from neural network
        with torch.no_grad():
            features = self.nn.predict_task_features(embedding)
            
        return features
        
    def load_or_init_nn(self):
        """Load existing neural network or initialize a new one."""
        model_path = f"models/agent_nn/{self.name}_nn.pt"
        try:
            self.nn.load_model(model_path)
            self.log_event(
                "nn_loaded",
                {"path": model_path}
            )
        except Exception as e:
            self.log_error(e, {
                "path": model_path,
                "action": "load_nn"
            })
            
    def save_nn(self):
        """Save neural network state."""
        model_path = f"models/agent_nn/{self.name}_nn.pt"
        try:
            self.nn.save_model(model_path)
            self.log_event(
                "nn_saved",
                {"path": model_path}
            )
        except Exception as e:
            self.log_error(e, {
                "path": model_path,
                "action": "save_nn"
            })
            
    def update_performance(self, task_metrics: TaskMetrics):
        """Update performance metrics.
        
        Args:
            task_metrics: Task execution metrics
        """
        self.nn.evaluate_performance(task_metrics)
        self.log_event(
            "performance_update",
            {
                "metrics": {
                    "response_time": task_metrics.response_time,
                    "confidence": task_metrics.confidence_score,
                    "user_feedback": task_metrics.user_feedback,
                    "task_success": task_metrics.task_success
                }
            }
        )
            
    async def shutdown(self):
        """Clean up resources."""
        if self.communication_hub:
            await self.communication_hub.deregister_agent(self.name)
        self.save_nn()

    def get_config(self) -> Dict[str, Any]:
        """Return a minimal configuration of the agent."""
        return {
            "name": self.name,
            "domain": self.name,
            "tools": [tool.name for tool in self.tools],
            "model_config": self.llm.get_config(),
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
        }

    def get_status(self) -> Dict[str, Any]:
        """Return basic runtime status."""
        metrics = self.nn.get_metrics()
        return {
            "agent_id": self.name,
            "name": self.name,
            "domain": self.name,
            "status": "active",
            "capabilities": [],
            "total_tasks": metrics.get("total_tasks", 0),
            "success_rate": metrics.get("success_rate", 0.0),
            "avg_response_time": metrics.get("avg_response_time", 0.0),
            "last_active": datetime.now().isoformat(),
        }
