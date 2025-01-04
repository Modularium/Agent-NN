"""Main CLI entry point."""
import click
from typing import Optional, List
from .core import SmolitCLI, console

@click.group()
def cli():
    """Smolit LLM-NN Command Line Interface."""
    pass

@cli.command()
@click.option("--username", prompt=True)
@click.option("--password", prompt=True, hide_input=True)
def login(username: str, password: str):
    """Authenticate with the system."""
    cli = SmolitCLI()
    cli.login(username, password)

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
    cli.submit_task(description, domain, priority, batch)

@cli.command()
@click.option("--backend", help="Backend type")
@click.option("--model", help="Model name")
@click.option("--system-prompt", help="System prompt")
def chat(backend: Optional[str],
        model: Optional[str],
        system_prompt: Optional[str]):
    """Start interactive chat session."""
    cli = SmolitCLI()
    cli.chat(backend, model, system_prompt)

@cli.command()
def agents():
    """List available agents."""
    cli = SmolitCLI()
    cli.list_agents()

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
    cli.create_agent(name, domain, capabilities, config_file)

@cli.command()
def metrics():
    """Display system metrics."""
    cli = SmolitCLI()
    cli.show_metrics()

@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def create_test(config_file: str):
    """Create a new A/B test."""
    cli = SmolitCLI()
    cli.create_test(config_file)

@cli.command()
@click.argument("test_id")
def test_results(test_id: str):
    """Show A/B test results."""
    cli = SmolitCLI()
    cli.show_test_results(test_id)

def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        console.print("[yellow]For help, run: smolit --help[/yellow]")