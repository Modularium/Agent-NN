"""Unified CLI for Smolit LLM-NN."""
import click
import asyncio
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .cli_app import CLIClient
from .conversation_manager import ConversationManager
from .llm_manager_cli import LLMManagerCLI
from .batch_processor import BatchProcessor
from .performance_monitor import PerformanceMonitor
from llm_models.llm_backend import LLMBackendType

console = Console()

class UnifiedCLI:
    """Unified CLI integrating all components."""
    
    def __init__(self):
        """Initialize unified CLI."""
        self.api_client = CLIClient()
        self.conversation_manager = ConversationManager()
        self.llm_manager = LLMManagerCLI()
        self.batch_processor = BatchProcessor()
        self.performance_monitor = PerformanceMonitor()
        
    def display_welcome(self):
        """Display welcome message."""
        welcome_md = """
        # 🤖 Smolit LLM-NN Unified CLI
        
        Welcome to the Smolit LLM-NN command-line interface!
        
        ## Available Commands:
        
        ### Chat & Conversation
        - `chat` - Start interactive chat session
        - `history` - Manage conversation history
        
        ### LLM Management
        - `llm list` - List available models
        - `llm switch` - Switch LLM backend
        - `llm setup` - Set up models
        - `llm test` - Test model functionality
        
        ### Batch Processing
        - `batch process` - Process batch of prompts
        - `batch status` - Check batch processing status
        
        ### Performance
        - `monitor live` - Show live performance metrics
        - `monitor report` - Generate performance report
        - `monitor compare` - Compare model performance
        
        ### System
        - `system status` - Show system status
        - `system config` - Manage configuration
        
        Use `--help` with any command for more information.
        """
        console.print(Markdown(welcome_md))

@click.group()
def cli():
    """Smolit LLM-NN Command Line Interface."""
    pass

# Chat commands
@cli.group()
def chat():
    """Chat and conversation commands."""
    pass

@chat.command()
@click.option("--backend", type=click.Choice([b.value for b in LLMBackendType]))
@click.option("--model", help="Model name")
@click.option("--system-prompt", help="System prompt")
def start(backend: Optional[str], model: Optional[str], system_prompt: Optional[str]):
    """Start interactive chat session."""
    manager = ConversationManager()
    if backend:
        manager.start_conversation(
            LLMBackendType(backend),
            model,
            system_prompt
        )
    else:
        manager.start_conversation()
    manager.chat()

@chat.command()
def history():
    """List conversation history."""
    manager = ConversationManager()
    manager.list_conversations()

# LLM management commands
@cli.group()
def llm():
    """LLM management commands."""
    pass

@llm.command()
@click.option("--backend", help="Filter by backend")
def list(backend: Optional[str]):
    """List available models."""
    manager = LLMManagerCLI()
    manager.list_models(backend)

@llm.command()
@click.argument("backend")
@click.option("--model", help="Model name")
def switch(backend: str, model: Optional[str]):
    """Switch LLM backend."""
    manager = LLMManagerCLI()
    manager.switch_backend(backend, model)

@llm.command()
@click.option("--llamafile", help="Llamafile model")
@click.option("--lmstudio", is_flag=True, help="Show LM Studio setup")
def setup(llamafile: Optional[str], lmstudio: bool):
    """Set up models."""
    manager = LLMManagerCLI()
    manager.setup_models(llamafile, lmstudio)

@llm.command()
@click.option("--backend", help="Backend to test")
@click.option("--prompt", default="Hello!", help="Test prompt")
def test(backend: Optional[str], prompt: str):
    """Test model functionality."""
    manager = LLMManagerCLI()
    manager.test_model(backend, prompt)

# Batch processing commands
@cli.group()
def batch():
    """Batch processing commands."""
    pass

@batch.command()
@click.argument("input_file")
@click.option("--backend", help="Backend to use")
@click.option("--model", help="Model name")
@click.option("--batch-size", default=10, help="Batch size")
@click.option("--output-format", type=click.Choice(["json", "csv"]), default="json")
def process(input_file: str,
           backend: Optional[str],
           model: Optional[str],
           batch_size: int,
           output_format: str):
    """Process batch of prompts."""
    processor = BatchProcessor()
    backend_type = LLMBackendType(backend) if backend else None
    asyncio.run(processor.process_file(
        input_file,
        backend_type,
        model,
        batch_size,
        output_format=output_format
    ))

# Performance monitoring commands
@cli.group()
def monitor():
    """Performance monitoring commands."""
    pass

@monitor.command()
def live():
    """Show live performance metrics."""
    monitor = PerformanceMonitor()
    monitor.show_live_metrics()

@monitor.command()
@click.option("--output", help="Output file path")
def report(output: Optional[str]):
    """Generate performance report."""
    monitor = PerformanceMonitor()
    monitor.generate_report(output)

@monitor.command()
@click.argument("logs", nargs=-1)
def compare(logs):
    """Compare model performance."""
    monitor = PerformanceMonitor()
    monitor.compare_models(logs)

# System commands
@cli.group()
def system():
    """System management commands."""
    pass

@system.command()
def status():
    """Show system status."""
    manager = LLMManagerCLI()
    manager.show_status()

@system.command()
@click.option("--show", is_flag=True, help="Show current config")
@click.option("--set", nargs=2, help="Set config value")
def config(show: bool, set):
    """Manage system configuration."""
    manager = LLMManagerCLI()
    if show:
        manager.manage_config({"show": True})
    elif set:
        manager.manage_config({"set": set})

def main():
    """Main entry point."""
    try:
        cli = UnifiedCLI()
        cli.display_welcome()
        cli()
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")

if __name__ == "__main__":
    main()