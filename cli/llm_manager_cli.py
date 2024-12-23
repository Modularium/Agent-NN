"""Command-line interface for managing LLM backends."""
import os
import sys
import argparse
import json
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import requests

from llm_models.llm_backend import LLMBackendType, LLMBackendManager
from config.llm_config import (
    LLAMAFILE_CONFIG,
    LMSTUDIO_CONFIG,
    OPENAI_CONFIG,
    get_model_config
)
from scripts.setup_local_models import (
    setup_llamafile,
    setup_lmstudio,
    verify_setup
)

console = Console()

class LLMManagerCLI:
    def __init__(self):
        """Initialize the CLI manager."""
        self.backend_manager = LLMBackendManager()
        
    def run(self):
        """Run the CLI application."""
        parser = argparse.ArgumentParser(
            description="Manage LLM backends and models"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # List command
        list_parser = subparsers.add_parser("list", help="List available models")
        list_parser.add_argument(
            "--backend",
            choices=[b.value for b in LLMBackendType],
            help="Filter by backend type"
        )
        
        # Switch command
        switch_parser = subparsers.add_parser("switch", help="Switch backend")
        switch_parser.add_argument(
            "backend",
            choices=[b.value for b in LLMBackendType],
            help="Backend to switch to"
        )
        switch_parser.add_argument(
            "--model",
            help="Specific model to use"
        )
        
        # Setup command
        setup_parser = subparsers.add_parser("setup", help="Set up models")
        setup_parser.add_argument(
            "--llamafile",
            choices=list(LLAMAFILE_CONFIG["models"].keys()) + ["all"],
            help="Llamafile model to set up"
        )
        setup_parser.add_argument(
            "--lmstudio",
            action="store_true",
            help="Show LM Studio setup instructions"
        )
        
        # Status command
        subparsers.add_parser("status", help="Show current status")
        
        # Test command
        test_parser = subparsers.add_parser("test", help="Test model functionality")
        test_parser.add_argument(
            "--backend",
            choices=[b.value for b in LLMBackendType],
            help="Backend to test"
        )
        test_parser.add_argument(
            "--prompt",
            default="Hello, how are you?",
            help="Test prompt to use"
        )
        
        # Config command
        config_parser = subparsers.add_parser("config", help="Manage configuration")
        config_parser.add_argument(
            "--show",
            action="store_true",
            help="Show current configuration"
        )
        config_parser.add_argument(
            "--set",
            nargs=2,
            metavar=("KEY", "VALUE"),
            help="Set configuration value"
        )
        
        args = parser.parse_args()
        
        try:
            if args.command == "list":
                self.list_models(args.backend)
            elif args.command == "switch":
                self.switch_backend(args.backend, args.model)
            elif args.command == "setup":
                self.setup_models(args.llamafile, args.lmstudio)
            elif args.command == "status":
                self.show_status()
            elif args.command == "test":
                self.test_model(args.backend, args.prompt)
            elif args.command == "config":
                self.manage_config(args)
            else:
                parser.print_help()
        except Exception as e:
            console.print(f"[red]Error: {str(e)}")
            sys.exit(1)
            
    def list_models(self, backend_type: Optional[str] = None):
        """List available models.
        
        Args:
            backend_type: Optional backend type to filter by
        """
        table = Table(title="Available Models")
        table.add_column("Backend")
        table.add_column("Model")
        table.add_column("Status")
        table.add_column("Description")
        
        models = self.backend_manager.get_available_models(
            LLMBackendType(backend_type) if backend_type else None
        )
        
        for backend, model_list in models.items():
            for model in model_list:
                config = get_model_config(backend, model)
                status = self._get_model_status(backend, model)
                table.add_row(
                    backend,
                    model,
                    status,
                    config.get("description", "")
                )
                
        console.print(table)
        
    def switch_backend(self, backend: str, model: Optional[str] = None):
        """Switch to a different backend.
        
        Args:
            backend: Backend type to switch to
            model: Optional specific model to use
        """
        try:
            backend_type = LLMBackendType(backend)
            self.backend_manager.set_backend(backend_type)
            
            if model:
                self.backend_manager.add_model(
                    backend_type,
                    model,
                    get_model_config(backend, model)
                )
                
            console.print(f"[green]Successfully switched to {backend}")
            if model:
                console.print(f"[green]Using model: {model}")
                
        except Exception as e:
            console.print(f"[red]Error switching backend: {str(e)}")
            
    def setup_models(self, llamafile: Optional[str], lmstudio: bool):
        """Set up models.
        
        Args:
            llamafile: Llamafile model to set up
            lmstudio: Whether to show LM Studio instructions
        """
        if llamafile:
            with Progress() as progress:
                task = progress.add_task(
                    f"Setting up {llamafile}...",
                    total=100
                )
                
                if llamafile == "all":
                    for model in LLAMAFILE_CONFIG["models"]:
                        progress.update(task, advance=50)
                        setup_llamafile(model)
                        progress.update(task, advance=50)
                else:
                    progress.update(task, advance=50)
                    setup_llamafile(llamafile)
                    progress.update(task, advance=50)
                    
        if lmstudio:
            console.print(Panel.fit(
                setup_lmstudio(),
                title="LM Studio Setup Instructions",
                border_style="blue"
            ))
            
    def show_status(self):
        """Show current system status."""
        # Get current backend info
        current_backend = self.backend_manager.current_backend
        
        # Create status table
        table = Table(title="System Status")
        table.add_column("Component")
        table.add_column("Status")
        table.add_column("Details")
        
        # Add backend status
        table.add_row(
            "Current Backend",
            current_backend.value,
            self._get_backend_status(current_backend)
        )
        
        # Add model status
        for backend_type in LLMBackendType:
            models = self.backend_manager.get_available_models(backend_type)
            for model in models.get(backend_type.value, []):
                status = self._get_model_status(backend_type.value, model)
                table.add_row(
                    f"{backend_type.value} Model",
                    model,
                    status
                )
                
        console.print(table)
        
    def test_model(self, backend: Optional[str], prompt: str):
        """Test model functionality.
        
        Args:
            backend: Optional backend to test
            prompt: Test prompt to use
        """
        if backend:
            self.backend_manager.set_backend(LLMBackendType(backend))
            
        try:
            llm = self.backend_manager.get_llm()
            
            with Progress() as progress:
                task = progress.add_task("Generating response...", total=100)
                
                # Generate response
                response = llm._call(prompt)
                progress.update(task, advance=100)
                
            # Display result
            console.print(Panel.fit(
                response,
                title="Model Response",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]Error testing model: {str(e)}")
            
    def manage_config(self, args):
        """Manage configuration settings.
        
        Args:
            args: Parsed command line arguments
        """
        if args.show:
            # Show current configuration
            config = {
                "openai": OPENAI_CONFIG,
                "lmstudio": LMSTUDIO_CONFIG,
                "llamafile": LLAMAFILE_CONFIG
            }
            
            console.print(Panel.fit(
                json.dumps(config, indent=2),
                title="Current Configuration",
                border_style="blue"
            ))
            
        elif args.set:
            key, value = args.set
            self._set_config(key, value)
            
    def _get_backend_status(self, backend_type: LLMBackendType) -> str:
        """Get status of a backend.
        
        Args:
            backend_type: Backend type to check
            
        Returns:
            str: Status description
        """
        try:
            if backend_type == LLMBackendType.OPENAI:
                return "Available" if OPENAI_CONFIG["api_key"] else "No API key"
                
            elif backend_type == LLMBackendType.LMSTUDIO:
                response = requests.get(LMSTUDIO_CONFIG["endpoint_url"])
                return "Running" if response.status_code == 200 else "Not running"
                
            elif backend_type == LLMBackendType.LLAMAFILE:
                return "Available" if os.path.exists(LLAMAFILE_CONFIG["models_dir"]) else "Not set up"
                
        except:
            return "Error checking status"
            
    def _get_model_status(self, backend: str, model: str) -> str:
        """Get status of a specific model.
        
        Args:
            backend: Backend type
            model: Model name
            
        Returns:
            str: Status description
        """
        try:
            if backend == "openai":
                return "Available"
                
            elif backend == "lmstudio":
                return "Available" if self._get_backend_status(LLMBackendType.LMSTUDIO) == "Running" else "Not running"
                
            elif backend == "llamafile":
                model_path = os.path.join(
                    LLAMAFILE_CONFIG["models_dir"],
                    LLAMAFILE_CONFIG["models"][model]["filename"]
                )
                return "Installed" if os.path.exists(model_path) else "Not installed"
                
        except:
            return "Error checking status"
            
    def _set_config(self, key: str, value: str):
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        # This is a placeholder for configuration management
        # In practice, you'd want to implement proper configuration persistence
        console.print(f"[yellow]Setting configuration not implemented yet")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Manage LLM backends and models"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Original commands
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--backend",
        choices=[b.value for b in LLMBackendType],
        help="Filter by backend type"
    )
    
    switch_parser = subparsers.add_parser("switch", help="Switch backend")
    switch_parser.add_argument(
        "backend",
        choices=[b.value for b in LLMBackendType],
        help="Backend to switch to"
    )
    switch_parser.add_argument(
        "--model",
        help="Specific model to use"
    )
    
    setup_parser = subparsers.add_parser("setup", help="Set up models")
    setup_parser.add_argument(
        "--llamafile",
        choices=list(LLAMAFILE_CONFIG["models"].keys()) + ["all"],
        help="Llamafile model to set up"
    )
    setup_parser.add_argument(
        "--lmstudio",
        action="store_true",
        help="Show LM Studio setup instructions"
    )
    
    subparsers.add_parser("status", help="Show current status")
    
    test_parser = subparsers.add_parser("test", help="Test model functionality")
    test_parser.add_argument(
        "--backend",
        choices=[b.value for b in LLMBackendType],
        help="Backend to test"
    )
    test_parser.add_argument(
        "--prompt",
        default="Hello, how are you?",
        help="Test prompt to use"
    )
    
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set configuration value"
    )
    
    # New commands for batch processing
    batch_parser = subparsers.add_parser("batch", help="Batch process prompts")
    batch_parser.add_argument(
        "input_file",
        help="Input file (CSV or JSON) containing prompts"
    )
    batch_parser.add_argument(
        "--backend",
        choices=[b.value for b in LLMBackendType],
        help="Backend to use"
    )
    batch_parser.add_argument(
        "--model",
        help="Specific model to use"
    )
    batch_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of prompts to process in parallel"
    )
    batch_parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default="json",
        help="Output format"
    )
    
    # New commands for conversation management
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--backend",
        choices=[b.value for b in LLMBackendType],
        help="Backend to use"
    )
    chat_parser.add_argument(
        "--model",
        help="Specific model to use"
    )
    chat_parser.add_argument(
        "--system-prompt",
        help="System prompt to use"
    )
    
    history_parser = subparsers.add_parser(
        "history",
        help="Manage conversation history"
    )
    history_parser.add_argument(
        "--list",
        action="store_true",
        help="List available conversations"
    )
    history_parser.add_argument(
        "--load",
        metavar="ID",
        help="Load a specific conversation"
    )
    
    # New commands for performance monitoring
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Performance monitoring"
    )
    monitor_parser.add_argument(
        "--live",
        action="store_true",
        help="Show live metrics"
    )
    monitor_parser.add_argument(
        "--compare",
        nargs="+",
        metavar="LOG",
        help="Compare performance across model logs"
    )
    monitor_parser.add_argument(
        "--report",
        metavar="FILE",
        help="Generate performance report"
    )
    
    args = parser.parse_args()
    
    if args.command == "batch":
        from .batch_processor import BatchProcessor
        import asyncio
        
        processor = BatchProcessor()
        asyncio.run(processor.process_file(
            args.input_file,
            LLMBackendType(args.backend) if args.backend else None,
            args.model,
            args.batch_size,
            output_format=args.output_format
        ))
        
    elif args.command == "chat":
        from .conversation_manager import ConversationManager
        
        manager = ConversationManager()
        manager.start_conversation(
            LLMBackendType(args.backend) if args.backend else None,
            args.model,
            args.system_prompt
        )
        manager.chat()
        
    elif args.command == "history":
        from .conversation_manager import ConversationManager
        
        manager = ConversationManager()
        if args.list:
            manager.list_conversations()
        elif args.load:
            manager.load_conversation(args.load)
            manager.chat()
            
    elif args.command == "monitor":
        from .performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        if args.live:
            monitor.show_live_metrics()
        elif args.compare:
            monitor.compare_models(args.compare)
        elif args.report:
            monitor.generate_report(args.report)
            
    else:
        cli = LLMManagerCLI()
        cli.run()

if __name__ == "__main__":
    main()