from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnablePassthrough
from utils.prompts import (
    CHATBOT_SYSTEM_PROMPT,
    TASK_IDENTIFICATION_PROMPT,
    TASK_RESULT_PROMPT,
    ERROR_HANDLING_PROMPT
)
from llm_models.llm_handler import get_llm_instance, get_default_llm

class ChatbotAgent:
    def __init__(self, supervisor_agent, llm_backend: str = "lmstudio"):
        """Initialize the chatbot agent.
        
        Args:
            supervisor_agent: The supervisor agent to delegate tasks to
            llm_backend: The LLM backend to use ("lmstudio" or "openai")
        """
        # Initialize LLM with different temperatures for different purposes
        self.conversation_llm = get_llm_instance(backend=llm_backend, temperature=0.7)
        self.task_llm = get_llm_instance(backend=llm_backend, temperature=0.2)
        
        # Create different chains for different purposes
        self.conversation_chain = RunnablePassthrough() | CHATBOT_SYSTEM_PROMPT | self.conversation_llm
        self.task_identification_chain = RunnablePassthrough() | TASK_IDENTIFICATION_PROMPT | self.task_llm
        self.result_formatting_chain = RunnablePassthrough() | TASK_RESULT_PROMPT | self.task_llm
        self.error_handling_chain = RunnablePassthrough() | ERROR_HANDLING_PROMPT | self.task_llm
        
        self.supervisor = supervisor_agent
        self.conversation_history: List[Dict[str, str]] = []
        
    def handle_user_message(self, user_message: str) -> str:
        """Handle a user message by either engaging in conversation or executing a task.
        
        Args:
            user_message: The message from the user
            
        Returns:
            str: The response to the user
        """
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Determine if message contains a task
            identification_result = self.task_identification_chain.invoke({
                "message": user_message
            }).content
            
            if identification_result.startswith("TASK:"):
                # Extract task description and execute it
                task_description = identification_result[5:].strip()
                execution_result = self.supervisor.execute_task(task_description)
                
                # Format the result
                if execution_result.get("success", False):
                    response = self.result_formatting_chain.invoke({
                        "task_description": task_description,
                        "result": execution_result["result"]
                    }).content
                else:
                    response = self.error_handling_chain.invoke({
                        "user_request": user_message,
                        "error_message": execution_result.get("error", "Unknown error occurred")
                    }).content
            else:
                # Handle as normal conversation
                chat_history = self._format_history()
                response = self.conversation_chain.invoke({
                    "input": user_message,
                    "chat_history": chat_history
                }).content
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            return response
            
        except Exception as e:
            # Handle any unexpected errors
            error_response = self.error_handling_chain.invoke({
                "user_request": user_message,
                "error_message": str(e)
            }).content
            
            self.conversation_history.append({
                "role": "assistant",
                "content": error_response
            })
            
            return error_response
    
    def _format_history(self) -> str:
        """Format conversation history for use in prompts."""
        formatted_history = []
        for msg in self.conversation_history[-10:]:  # Keep last 10 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        return "\n".join(formatted_history)
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation.
        
        Returns:
            Dict containing conversation statistics and metadata
        """
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for msg in self.conversation_history if msg["role"] == "user"),
            "assistant_messages": sum(1 for msg in self.conversation_history if msg["role"] == "assistant"),
            "last_message_timestamp": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
