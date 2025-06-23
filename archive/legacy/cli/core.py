"""Core CLI implementation for Agent-NN."""
import os
import json
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt

from llm_models.llm_backend import LLMBackendManager, LLMBackendType
from agents.chatbot_agent import ChatbotAgent
from agents.supervisor_agent import SupervisorAgent
from .batch_processor import BatchProcessor
from .performance_monitor import PerformanceMonitor
from utils.logging_util import LoggerMixin

console = Console()

class SmolitCLI(LoggerMixin):
    """Core CLI class integrating all functionality."""
    
    def __init__(self):
        """Initialize CLI components."""
        super().__init__()
        self.config = self._load_config()
        self.backend_manager = LLMBackendManager()
        self.batch_processor = BatchProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.supervisor = None
        self.chatbot = None
        
    def _load_config(self) -> dict:
        """Load configuration from file or environment."""
        config = {
            "api_url": os.getenv("SMOLIT_API_URL", "http://localhost:8000"),
            "token_file": os.getenv("SMOLIT_TOKEN_FILE", "~/.smolit/token"),
            "default_priority": 1,
            "output_format": "rich"
        }
        
        config_file = os.path.expanduser("~/.smolit/config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                config.update(json.load(f))
                
        return config
        
    def initialize_agents(self):
        """Initialize required agents."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Initialize supervisor
            task = progress.add_task("Initializing supervisor agent...", total=None)
            self.supervisor = SupervisorAgent()
            progress.update(task, completed=True)
            
            # Initialize chatbot
            task = progress.add_task("Initializing chatbot agent...", total=None)
            self.chatbot = ChatbotAgent(self.supervisor)
            progress.update(task, completed=True)
            
    def login(self, username: str, password: str):
        """Authenticate user.
        
        Args:
            username: Username
            password: Password
        """
        token_dir = os.path.dirname(os.path.expanduser(self.config["token_file"]))
        os.makedirs(token_dir, exist_ok=True)
        
        # Here you would implement actual authentication
        # For now, just store a dummy token
        with open(os.path.expanduser(self.config["token_file"]), "w") as f:
            f.write(f"dummy_token_{username}")
            
        console.print("[green]Login successful")
        
    def submit_task(self,
                   description: str,
                   domain: Optional[str] = None,
                   priority: int = 1,
                   batch_file: Optional[str] = None):
        """Submit task(s) for execution.
        
        Args:
            description: Task description
            domain: Optional domain hint
            priority: Task priority
            batch_file: Optional batch file path
        """
        if not self.supervisor:
            self.initialize_agents()
            
        if batch_file:
            # Process batch file
            with open(batch_file) as f:
                batch_data = json.load(f)
                
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(
                    "Processing batch tasks...",
                    total=len(batch_data["tasks"])
                )
                
                for task_data in batch_data["tasks"]:
                    result = self.supervisor.execute_task(
                        task_data["description"],
                        domain=task_data.get("domain"),
                        priority=task_data.get("priority", 1)
                    )
                    self._display_task_result(result)
                    progress.advance(task)
                    
        else:
            # Process single task
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing task...", total=None)
                result = self.supervisor.execute_task(
                    description,
                    domain=domain,
                    priority=priority
                )
                progress.update(task, completed=True)
                
            self._display_task_result(result)
            
    def chat(self,
            backend: Optional[str] = None,
            model: Optional[str] = None,
            system_prompt: Optional[str] = None):
        """Start interactive chat session.
        
        Args:
            backend: Optional backend type
            model: Optional model name
            system_prompt: Optional system prompt
        """
        if not self.chatbot:
            self.initialize_agents()
            
        # Set up backend if specified
        if backend:
            self.backend_manager.set_backend(LLMBackendType(backend))
            if model:
                self.backend_manager.add_model(
                    LLMBackendType(backend),
                    model,
                    {}
                )
                
        # Display welcome message
        welcome_md = """
        # 🤖 Smolit Chat

        Welcome to interactive chat mode!
        
        Commands:
        - /help - Show help
        - /clear - Clear history
        - /save - Save conversation
        - /quit - Exit chat
        """
        console.print(Markdown(welcome_md))
        
        # Start chat loop
        try:
            while True:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['/quit', '/exit']:
                    break
                    
                if user_input.startswith('/'):
                    self._handle_chat_command(user_input)
                    continue
                    
                # Get response
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Thinking...", total=None)
                    response = self.chatbot.handle_user_message(user_input)
                    progress.update(task, completed=True)
                    
                console.print("\n[bold green]Assistant:[/bold green]", response)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat session ended[/yellow]")
            
    def list_agents(self):
        """List available agents."""
        if not self.supervisor:
            self.initialize_agents()
            
        table = Table(title="Available Agents")
        table.add_column("Name")
        table.add_column("Domain")
        table.add_column("Status")
        table.add_column("Capabilities")
        
        for agent in self.supervisor.list_agents():
            table.add_row(
                agent["name"],
                agent["domain"],
                agent["status"],
                ", ".join(agent["capabilities"])
            )
            
        console.print(table)
        
    def create_agent(self,
                    name: str,
                    domain: str,
                    capabilities: List[str],
                    config_file: str):
        """Create a new agent.
        
        Args:
            name: Agent name
            domain: Agent domain
            capabilities: Agent capabilities
            config_file: Configuration file path
        """
        if not self.supervisor:
            self.initialize_agents()
            
        # Load configuration
        with open(config_file) as f:
            config = json.load(f)
            
        # Create agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating agent...", total=None)
            self.supervisor.create_agent(name, domain, capabilities, config)
            progress.update(task, completed=True)
            
        console.print(f"[green]Agent '{name}' created successfully")
        
    def show_metrics(self):
        """Display system metrics."""
        metrics = self.performance_monitor.get_summary()
        
        if not metrics:
            console.print("[yellow]No metrics available")
            return
            
        table = Table(title="System Metrics")
        table.add_column("Metric")
        table.add_column("Value")
        
        for metric, value in metrics.items():
            table.add_row(
                metric.replace("_", " ").title(),
                f"{value:.2f}" if isinstance(value, float) else str(value)
            )
            
        console.print(table)
        
    def create_test(self, config_file: str):
        """Create A/B test.
        
        Args:
            config_file: Test configuration file
        """
        # Load configuration
        with open(config_file) as f:
            config = json.load(f)
            
        # Create test
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Here you would implement actual test creation
        console.print(f"[green]Test created with ID: {test_id}")
        
    def show_test_results(self, test_id: str):
        """Show A/B test results.
        
        Args:
            test_id: Test identifier
        """
        # Here you would implement actual test results retrieval
        results = {
            "status": "completed",
            "variants": {
                "baseline": {"success_rate": 0.85},
                "experimental": {"success_rate": 0.92}
            },
            "winner": "experimental",
            "confidence": 0.95
        }
        
        console.print(Panel(
            Syntax(json.dumps(results, indent=2), "json", theme="monokai"),
            title=f"Test Results: {test_id}"
        ))
        
    def _display_task_result(self, result: Dict[str, Any]):
        """Display task execution result.
        
        Args:
            result: Task result dictionary
        """
        # Display result
        if result.get("success"):
            console.print("\n[bold]Task Result:[/bold]")
            console.print(Panel(
                Syntax(
                    json.dumps(result["result"], indent=2),
                    "json",
                    theme="monokai"
                )
            ))
            
            # Display metrics
            if "metrics" in result:
                table = Table(title="Performance Metrics")
                table.add_column("Metric")
                table.add_column("Value")
                
                for metric, value in result["metrics"].items():
                    table.add_row(metric, f"{value:.4f}")
                    
                console.print(table)
                
        else:
            console.print(f"[red]Error: {result.get('error', 'Unknown error')}")
            
    def _handle_chat_command(self, command: str):
        """Handle chat commands.
        
        Args:
            command: Command string
        """
        cmd = command.lower()
        
        if cmd == '/help':
            help_text = """
            Available Commands:
            ------------------
            /help   - Show this help message
            /clear  - Clear conversation history
            /save   - Save conversation
            /quit   - End conversation
            """
            console.print(Panel(help_text, title="Help"))
            
        elif cmd == '/clear':
            self.chatbot.clear_history()
            console.print("[green]Conversation history cleared")
            
        elif cmd == '/save':
            # Save conversation history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump(self.chatbot.get_history(), f, indent=2)
                
            console.print(f"[green]Conversation saved to {filename}")
            
        else:
            console.print("[yellow]Unknown command. Type /help for available commands")
