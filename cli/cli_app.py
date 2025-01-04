import click
import asyncio
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.syntax import Syntax
from rich.panel import Panel

class CLIClient:
    """CLI client for Smolit LLM-NN."""
    
    def __init__(self,
                 api_url: str = "http://localhost:8000",
                 token_file: str = "~/.smolit/token"):
        """Initialize client.
        
        Args:
            api_url: API server URL
            token_file: Token file path
        """
        self.api_url = api_url
        self.token_file = os.path.expanduser(token_file)
        self.console = Console()
        
        # Load token
        self.token = self._load_token()
        
    def _load_token(self) -> Optional[str]:
        """Load authentication token.
        
        Returns:
            Optional[str]: Authentication token
        """
        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                return f.read().strip()
        return None
        
    def _save_token(self, token: str):
        """Save authentication token.
        
        Args:
            token: Authentication token
        """
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        with open(self.token_file, "w") as f:
            f.write(token)
            
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers.
        
        Returns:
            Dict[str, str]: Request headers
        """
        if not self.token:
            raise click.ClickException("Not authenticated. Run 'login' first.")
            
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
    def login(self, username: str, password: str):
        """Login to API server.
        
        Args:
            username: Username
            password: Password
        """
        try:
            response = requests.post(
                f"{self.api_url}/token",
                data={
                    "username": username,
                    "password": password
                }
            )
            response.raise_for_status()
            
            # Save token
            token = response.json()["access_token"]
            self._save_token(token)
            self.token = token
            
            self.console.print("[green]Login successful[/green]")
            
        except Exception as e:
            raise click.ClickException(f"Login failed: {str(e)}")
            
    def submit_task(self,
                   description: str,
                   domain: Optional[str] = None,
                   priority: int = 1):
        """Submit task for execution.
        
        Args:
            description: Task description
            domain: Optional domain hint
            priority: Task priority
        """
        try:
            # Create request
            data = {
                "description": description,
                "domain": domain,
                "priority": priority
            }
            
            with Progress() as progress:
                task = progress.add_task(
                    "Executing task...",
                    total=None
                )
                
                # Submit task
                response = requests.post(
                    f"{self.api_url}/tasks",
                    headers=self._get_headers(),
                    json=data
                )
                response.raise_for_status()
                
                result = response.json()
                
            # Display result
            self.console.print("\n[bold]Task Result:[/bold]")
            self.console.print(Panel(
                Syntax(
                    json.dumps(result["result"], indent=2),
                    "json",
                    theme="monokai"
                )
            ))
            
            # Display metrics
            table = Table(title="Performance Metrics")
            table.add_column("Metric")
            table.add_column("Value")
            
            for metric, value in result["metrics"].items():
                table.add_row(metric, f"{value:.4f}")
                
            self.console.print(table)
            
        except Exception as e:
            raise click.ClickException(f"Task execution failed: {str(e)}")
            
    def list_agents(self):
        """List available agents."""
        try:
            response = requests.get(
                f"{self.api_url}/agents",
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            agents = response.json()
            
            # Display agents
            table = Table(title="Available Agents")
            table.add_column("Name")
            table.add_column("Domain")
            table.add_column("Capabilities")
            
            for agent in agents:
                table.add_row(
                    agent["name"],
                    agent["domain"],
                    ", ".join(agent["capabilities"])
                )
                
            self.console.print(table)
            
        except Exception as e:
            raise click.ClickException(f"Failed to list agents: {str(e)}")
            
    def show_metrics(self):
        """Show system metrics."""
        try:
            response = requests.get(
                f"{self.api_url}/metrics",
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            metrics = response.json()
            
            # Display metrics
            table = Table(title="System Metrics")
            table.add_column("Metric")
            table.add_column("Value")
            
            for metric, value in metrics.items():
                table.add_row(
                    metric.replace("_", " ").title(),
                    f"{value:.2f}"
                )
                
            self.console.print(table)
            
        except Exception as e:
            raise click.ClickException(f"Failed to get metrics: {str(e)}")
            
    def create_agent(self,
                    name: str,
                    domain: str,
                    capabilities: List[str],
                    config_file: str):
        """Create new agent.
        
        Args:
            name: Agent name
            domain: Agent domain
            capabilities: Agent capabilities
            config_file: Configuration file path
        """
        try:
            # Load configuration
            with open(config_file, "r") as f:
                config = json.load(f)
                
            # Create request
            data = {
                "name": name,
                "domain": domain,
                "capabilities": capabilities,
                "config": config
            }
            
            response = requests.post(
                f"{self.api_url}/agents",
                headers=self._get_headers(),
                json=data
            )
            response.raise_for_status()
            
            self.console.print("[green]Agent created successfully[/green]")
            
        except Exception as e:
            raise click.ClickException(f"Failed to create agent: {str(e)}")
            
    def create_test(self, config_file: str):
        """Create A/B test.
        
        Args:
            config_file: Test configuration file path
        """
        try:
            # Load configuration
            with open(config_file, "r") as f:
                config = json.load(f)
                
            response = requests.post(
                f"{self.api_url}/tests",
                headers=self._get_headers(),
                json=config
            )
            response.raise_for_status()
            
            result = response.json()
            self.console.print(
                f"[green]Test created with ID: {result['test_id']}[/green]"
            )
            
        except Exception as e:
            raise click.ClickException(f"Failed to create test: {str(e)}")
            
    def show_test_results(self, test_id: str):
        """Show A/B test results.
        
        Args:
            test_id: Test identifier
        """
        try:
            response = requests.get(
                f"{self.api_url}/tests/{test_id}",
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Display results
            self.console.print("\n[bold]Test Results:[/bold]")
            self.console.print(Panel(
                Syntax(
                    json.dumps(results, indent=2),
                    "json",
                    theme="monokai"
                )
            ))
            
        except Exception as e:
            raise click.ClickException(
                f"Failed to get test results: {str(e)}"
            )

# CLI Commands
@click.group()
def cli():
    """Smolit LLM-NN Command Line Interface."""
    pass

@cli.command()
@click.option("--username", prompt=True)
@click.option("--password", prompt=True, hide_input=True)
def login(username: str, password: str):
    """Login to API server."""
    client = CLIClient()
    client.login(username, password)

@cli.command()
@click.argument("description")
@click.option("--domain", help="Optional domain hint")
@click.option("--priority", default=1, help="Task priority (1-10)")
def task(description: str, domain: Optional[str], priority: int):
    """Submit task for execution."""
    client = CLIClient()
    client.submit_task(description, domain, priority)

@cli.command()
def agents():
    """List available agents."""
    client = CLIClient()
    client.list_agents()

@cli.command()
def metrics():
    """Show system metrics."""
    client = CLIClient()
    client.show_metrics()

@cli.command()
@click.argument("name")
@click.argument("domain")
@click.argument("capabilities", nargs=-1)
@click.argument("config_file", type=click.Path(exists=True))
def create_agent(name: str,
                domain: str,
                capabilities: List[str],
                config_file: str):
    """Create new agent."""
    client = CLIClient()
    client.create_agent(name, domain, capabilities, config_file)

@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def create_test(config_file: str):
    """Create A/B test."""
    client = CLIClient()
    client.create_test(config_file)

@cli.command()
@click.argument("test_id")
def test_results(test_id: str):
    """Show A/B test results."""
    client = CLIClient()
    client.show_test_results(test_id)

if __name__ == "__main__":
    cli()