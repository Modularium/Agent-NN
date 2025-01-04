from typing import Dict, Any, Optional, List, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.base import BaseTool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction, AgentFinish, Document
import json
import aiohttp
import asyncio
from datetime import datetime
from utils.logging_util import LoggerMixin
from config.llm_config import OPENAI_CONFIG

class AgenticWorker(LoggerMixin):
    """Enhanced worker agent with LangChain agentic features."""
    
    def __init__(self,
                 name: str,
                 domain: str,
                 tools: Optional[List[BaseTool]] = None,
                 domain_docs: Optional[List[Document]] = None):
        """Initialize agentic worker.
        
        Args:
            name: Agent name
            domain: Agent domain
            tools: Optional list of tools
            domain_docs: Optional domain knowledge
        """
        super().__init__()
        self.name = name
        self.domain = domain
        
        # Initialize tools
        self.tools = tools or self._get_default_tools()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize agent components
        self.prompt = self._create_prompt()
        self.output_parser = ReActSingleInputOutputParser()
        
        # Create agent executor
        self.agent = LLMSingleActionAgent(
            llm_chain=self._create_llm_chain(),
            output_parser=self.output_parser,
            stop=["\nObservation:", "\nThought:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
    def _get_default_tools(self) -> List[BaseTool]:
        """Get default tools for agent.
        
        Returns:
            List[BaseTool]: Default tools
        """
        return [
            Tool(
                name="search_knowledge",
                func=self._search_knowledge,
                description="Search domain knowledge base"
            ),
            Tool(
                name="calculate",
                func=self._calculate,
                description="Perform calculations"
            ),
            Tool(
                name="api_request",
                func=self._api_request,
                description="Make API requests"
            ),
            Tool(
                name="analyze_data",
                func=self._analyze_data,
                description="Analyze structured data"
            )
        ]
        
    def _create_prompt(self) -> PromptTemplate:
        """Create agent prompt template.
        
        Returns:
            PromptTemplate: Prompt template
        """
        template = """You are a specialized agent for {domain} tasks.

Available Tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=[
                "domain",
                "tools",
                "tool_names",
                "chat_history",
                "input",
                "agent_scratchpad"
            ]
        )
        
    def _create_llm_chain(self):
        """Create LLM chain for agent.
        
        Returns:
            LLMChain: Agent LLM chain
        """
        from langchain.chat_models import ChatOpenAI
        from langchain.chains import LLMChain
        
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=OPENAI_CONFIG["api_key"]
        )
        
        return LLMChain(
            llm=llm,
            prompt=self.prompt
        )
        
    async def _search_knowledge(self, query: str) -> str:
        """Search domain knowledge.
        
        Args:
            query: Search query
            
        Returns:
            str: Search results
        """
        # Implement knowledge base search
        return "Knowledge base results for: " + query
        
    async def _calculate(self, expression: str) -> str:
        """Perform calculations.
        
        Args:
            expression: Math expression
            
        Returns:
            str: Calculation result
        """
        try:
            # Safely evaluate math expression
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
            
    async def _api_request(self, request: Dict[str, Any]) -> str:
        """Make API request.
        
        Args:
            request: Request details
            
        Returns:
            str: API response
        """
        try:
            method = request.get("method", "GET")
            url = request.get("url")
            headers = request.get("headers", {})
            data = request.get("data")
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.text()
                    return result
                    
        except Exception as e:
            return f"API Error: {str(e)}"
            
    async def _analyze_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """Analyze structured data.
        
        Args:
            data: Data to analyze
            
        Returns:
            str: Analysis results
        """
        try:
            if isinstance(data, str):
                data = json.loads(data)
                
            # Implement data analysis
            analysis = {
                "type": type(data).__name__,
                "size": len(data),
                "summary": "Data analysis results"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Analysis Error: {str(e)}"
            
    async def execute_task(self,
                          task_description: str,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute task using agent tools.
        
        Args:
            task_description: Task description
            context: Optional context
            
        Returns:
            Dict[str, Any]: Task results
        """
        try:
            # Add context to memory
            if context:
                self.memory.save_context(
                    {"input": "Context"},
                    {"output": json.dumps(context)}
                )
                
            # Execute task
            start_time = datetime.now()
            result = await self.executor.arun(
                input=task_description,
                domain=self.domain
            )
            duration = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response = {
                "result": result,
                "execution_time": duration,
                "memory_size": len(self.memory.chat_memory.messages)
            }
            
            # Log execution
            self.log_event(
                "task_executed",
                {
                    "task": task_description,
                    "duration": duration,
                    "success": True
                }
            )
            
            return response
            
        except Exception as e:
            self.log_error(e, {
                "task": task_description,
                "context": context
            })
            
            return {
                "error": str(e),
                "execution_time": 0,
                "memory_size": len(self.memory.chat_memory.messages)
            }
            
    def add_tool(self, tool: BaseTool):
        """Add new tool to agent.
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        self.executor.tools = self.tools
        self.agent.allowed_tools = [tool.name for tool in self.tools]
        
        # Log tool addition
        self.log_event(
            "tool_added",
            {"tool": tool.name}
        )
        
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Optional[BaseTool]: Tool or None
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
        
    def get_memory_contents(self) -> List[Dict[str, Any]]:
        """Get agent memory contents.
        
        Returns:
            List[Dict[str, Any]]: Memory messages
        """
        return [
            {
                "type": msg.type,
                "content": msg.content,
                "timestamp": msg.additional_kwargs.get("timestamp")
            }
            for msg in self.memory.chat_memory.messages
        ]
        
    def clear_memory(self):
        """Clear agent memory."""
        self.memory.clear()
        self.log_event("memory_cleared", {})