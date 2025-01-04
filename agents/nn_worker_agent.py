import time
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
from nn_models.agent_nn import AgentNN, TaskMetrics
import torch
import os

class NNWorkerAgent:
    def __init__(self,
                 name: str,
                 domain_docs: Optional[Union[List[str], List[Document]]] = None,
                 communication_hub: Optional[AgentCommunicationHub] = None,
                 knowledge_manager: Optional[DomainKnowledgeManager] = None):
        """Initialize a worker agent with neural network enhancement.
        
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
        
        # Initialize neural network for task optimization
        self.nn = AgentNN()
        self.load_or_init_nn()
        
        # Create a custom prompt template for the QA chain
        qa_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        If you need information from another domain, indicate that in your response.
        
        Context: {context}
        
        Task-specific features:
        {task_features}
        
        Question: {question}
        
        Answer:"""
        
        qa_prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context", "question", "task_features"]
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
            
    def get_task_embedding(self, task_description: str) -> torch.Tensor:
        """Get embedding for task description.
        
        Args:
            task_description: Task description
            
        Returns:
            torch.Tensor: Task embedding
        """
        # Use LLM's embedding function
        embedding = self.llm.get_embedding(task_description)
        return torch.tensor(embedding).unsqueeze(0)  # Add batch dimension
        
    def format_task_features(self, features: torch.Tensor) -> str:
        """Format task features for prompt.
        
        Args:
            features: Task features tensor
            
        Returns:
            str: Formatted features string
        """
        # Convert features to list and format as string
        features_list = features.squeeze().tolist()
        return "\n".join([f"Feature {i+1}: {val:.3f}" for i, val in enumerate(features_list)])
        
    def load_or_init_nn(self) -> None:
        """Load existing neural network or initialize a new one."""
        model_path = f"models/agent_nn/{self.name}_nn.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            try:
                self.nn.load_model(model_path)
            except Exception as e:
                print(f"Error loading model for {self.name}: {e}")
                print("Initializing new neural network.")
        
    def save_nn(self) -> None:
        """Save neural network state."""
        model_path = f"models/agent_nn/{self.name}_nn.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.nn.save_model(model_path)
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent's performance metrics.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        training_summary = self.nn.get_training_summary()
        
        # Calculate average metrics from recent evaluations
        recent_evals = self.nn.eval_metrics[-100:]  # Last 100 evaluations
        if recent_evals:
            avg_metrics = {
                "avg_response_time": sum(m["response_time"] for m in recent_evals) / len(recent_evals),
                "avg_confidence": sum(m["confidence"] for m in recent_evals) / len(recent_evals)
            }
            
            # Include user feedback if available
            feedback_scores = [m["user_feedback"] for m in recent_evals if "user_feedback" in m]
            if feedback_scores:
                avg_metrics["avg_user_feedback"] = sum(feedback_scores) / len(feedback_scores)
            
            # Include success rate if available
            success_rates = [m["success_rate"] for m in recent_evals if "success_rate" in m]
            if success_rates:
                avg_metrics["success_rate"] = sum(success_rates) / len(success_rates)
                
            training_summary.update(avg_metrics)
            
        return training_summary
        
    async def execute_task(self,
                          task_description: str,
                          context: Optional[str] = None) -> str:
        """Execute a task using the agent's knowledge, capabilities, and neural network.
        
        Args:
            task_description: Description of the task to execute
            context: Optional additional context for the task
            
        Returns:
            str: The agent's response to the task
        """
        start_time = time.time()
        
        try:
            # Get task embedding and optimize task features
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
                            
            # Execute task with task features
            if context:
                response, confidence = self.llm.generate_with_confidence(
                    task_description,
                    context=context,
                    task_features=task_features_str
                )
            else:
                response, confidence = self.llm.generate_with_confidence(
                    task_description,
                    task_features=task_features_str
                )
                
            # Record task metrics
            elapsed_time = time.time() - start_time
            metrics = TaskMetrics(
                response_time=elapsed_time,
                confidence_score=confidence
            )
            self.nn.evaluate_performance(metrics)
            
            return response
                
        except Exception as e:
            # Log error and return error message
            logger.error(f"Error executing task: {str(e)}")
            return f"Error executing task: {str(e)}"
            
    # Inherit other methods from WorkerAgent
    def ingest_knowledge(self, documents: Union[List[str], List[Document]], metadata: dict = None) -> None:
        """Add new documents to the agent's knowledge base."""
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
                
    def search_knowledge_base(self, query: str, k: int = 4) -> List[Document]:
        """Search the agent's knowledge base directly."""
        return self.db.search(query, k=k)
        
    def clear_knowledge(self) -> None:
        """Clear all documents from the agent's knowledge base."""
        self.db.clear_knowledge_base()
        
    async def shutdown(self):
        """Clean up resources and save neural network state."""
        if self.communication_hub:
            await self.communication_hub.deregister_agent(self.name)
        self.save_nn()