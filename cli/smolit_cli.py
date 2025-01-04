"""Smolit LLM-NN Command Line Interface."""
import os
import json
import click
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from .cli_app import CLIClient
from .batch_processor import BatchProcessor
from .performance_monitor import PerformanceMonitor
from utils.logging_util import LoggerMixin

console = Console()

class SmolitCLI(LoggerMixin):
    """Main CLI class implementing the Smolit CLI reference."""
    
    def __init__(self):
        """Initialize CLI components."""
        super().__init__()
        self.config = self._load_config()
        self.client = CLIClient(
            api_url=self.config.get("api_url", "http://localhost:8000"),
            token_file=self.config.get("token_file", "~/.smolit/token")
        )
        self.batch_processor = BatchProcessor()
        self.performance_monitor = PerformanceMonitor()
        
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

@click.group()
def cli():
    """Smolit LLM-NN Command Line Interface."""
    pass

@cli.command()
@click.option("--username", prompt=True)
@click.option("--password", prompt=True, hide_input=True)
def login(username: str, password: str):
    """Authenticate with the API server."""
    cli = SmolitCLI()
    cli.client.login(username, password)

@cli.command()
@click.argument("description")
@click.option("--domain", help="Optional domain hint")
@click.option("--priority", type=int, default=1, help="Task priority (1-10)")
@click.option("--batch", type=click.Path(exists=True), help="Batch task file")
def task(description: str,
        domain: Optional[str],
        priority: int,
        batch: Optional[str]):
    """Submit task(s) for execution."""
    cli = SmolitCLI()
    
    if batch:
        # Process batch file
        with open(batch) as f:
            batch_data = json.load(f)
            for task_data in batch_data["tasks"]:
                cli.client.submit_task(
                    task_data["description"],
                    task_data.get("domain"),
                    task_data.get("priority", 1)
                )
    else:
        # Process single task
        cli.client.submit_task(description, domain, priority)

@cli.command()
def agents():
    """List available agents."""
    cli = SmolitCLI()
    cli.client.list_agents()

@cli.command()
@click.argument("name")
@click.argument("domain")
@click.argument("capabilities", nargs=-1)
@click.argument("config_file", type=click.Path(exists=True))
def create_agent(name: str,
                domain: str,
                capabilities: List[str],
                config_file: str):
    """Create a new agent."""
    cli = SmolitCLI()
    cli.client.create_agent(name, domain, capabilities, config_file)

@cli.command()
def metrics():
    """Display system metrics."""
    cli = SmolitCLI()
    cli.client.show_metrics()

@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def create_test(config_file: str):
    """Create a new A/B test."""
    cli = SmolitCLI()
    cli.client.create_test(config_file)

@cli.command()
@click.argument("test_id")
def test_results(test_id: str):
    """Show A/B test results."""
    cli = SmolitCLI()
    cli.client.show_test_results(test_id)

def format_error(error_type: str, description: str, resolution: str):
    """Format error message.
    
    Args:
        error_type: Type of error
        description: Error description
        resolution: Suggested resolution
    """
    console.print(f"[red]Error:[/red] {error_type}")
    console.print(f"[yellow]Cause:[/yellow] {description}")
    console.print(f"[green]Resolution:[/green] {resolution}")

def format_json(data: dict) -> Panel:
    """Format JSON data in a panel.
    
    Args:
        data: Data to format
        
    Returns:
        Panel: Formatted panel
    """
    return Panel(
        Syntax(
            json.dumps(data, indent=2),
            "json",
            theme="monokai"
        ),
        title="Result"
    )

def format_table(title: str, data: List[dict]) -> Table:
    """Format data in a table.
    
    Args:
        title: Table title
        data: List of dictionaries
        
    Returns:
        Table: Formatted table
    """
    table = Table(title=title)
    
    # Add columns
    if data:
        for key in data[0].keys():
            table.add_column(key.replace("_", " ").title())
            
        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row.values()])
            
    return table

def show_progress(description: str):
    """Show progress indicator.
    
    Args:
        description: Progress description
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[progress.description]{description}"),
        console=console
    )

def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        format_error(
            type(e).__name__,
            str(e),
            "Check the documentation or contact support"
        )