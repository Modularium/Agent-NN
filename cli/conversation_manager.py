"""Conversation management for interactive LLM sessions."""
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from llm_models.llm_backend import LLMBackendManager, LLMBackendType
from .performance_monitor import PerformanceMonitor

console = Console()

class ConversationManager:
    def __init__(self, history_dir: str = "history/conversations"):
        """Initialize conversation manager.
        
        Args:
            history_dir: Directory for conversation history
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.backend_manager = LLMBackendManager()
        self.performance_monitor = PerformanceMonitor()
        
        self.current_conversation = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "messages": [],
            "metadata": {}
        }
        
    def start_conversation(self,
                         backend_type: Optional[LLMBackendType] = None,
                         model_name: Optional[str] = None,
                         system_prompt: Optional[str] = None):
        """Start a new conversation.
        
        Args:
            backend_type: Optional backend to use
            model_name: Optional specific model to use
            system_prompt: Optional system prompt
        """
        # Set up backend
        if backend_type:
            self.backend_manager.set_backend(backend_type)
        if model_name:
            self.backend_manager.add_model(
                self.backend_manager.current_backend,
                model_name,
                {}
            )
            
        # Initialize conversation
        self.current_conversation["metadata"].update({
            "backend": self.backend_manager.current_backend.value,
            "model": model_name or "default",
            "system_prompt": system_prompt
        })
        
        if system_prompt:
            self.current_conversation["messages"].append({
                "role": "system",
                "content": system_prompt,
                "timestamp": datetime.now().isoformat()
            })
            
        # Start performance monitoring
        self.performance_monitor.start_monitoring(
            model_name or "default",
            self.backend_manager.current_backend.value
        )
        
        self._show_conversation_start()
        
    def chat(self):
        """Start interactive chat session."""
        try:
            while True:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['/quit', '/exit']:
                    break
                    
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                    
                # Add user message to history
                self.add_message("user", user_input)
                
                # Get model response
                try:
                    start_time = datetime.now().timestamp()
                    response = self._get_response(user_input)
                    end_time = datetime.now().timestamp()
                    
                    # Log performance
                    self.performance_monitor.log_inference(
                        user_input,
                        response,
                        start_time,
                        end_time
                    )
                    
                    # Add response to history
                    self.add_message("assistant", response)
                    
                    # Display response
                    self._display_response(response)
                    
                except Exception as e:
                    console.print(f"[red]Error: {str(e)}")
                    
        finally:
            self._save_conversation()
            self.performance_monitor.save_session()
            
    def add_message(self, role: str, content: str):
        """Add a message to the conversation.
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
        """
        self.current_conversation["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
    def load_conversation(self, conversation_id: str):
        """Load a previous conversation.
        
        Args:
            conversation_id: ID of conversation to load
        """
        filepath = self.history_dir / f"{conversation_id}.json"
        try:
            with open(filepath, 'r') as f:
                self.current_conversation = json.load(f)
                self._show_conversation_history()
        except Exception as e:
            console.print(f"[red]Error loading conversation: {str(e)}")
            
    def list_conversations(self):
        """List available conversation histories."""
        conversations = []
        for file in self.history_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    conversations.append({
                        "id": data["id"],
                        "start_time": data["start_time"],
                        "messages": len(data["messages"]),
                        "backend": data["metadata"].get("backend", "unknown"),
                        "model": data["metadata"].get("model", "unknown")
                    })
            except:
                continue
                
        if not conversations:
            console.print("[yellow]No conversation histories found")
            return
            
        # Sort by start time
        conversations.sort(key=lambda x: x["start_time"], reverse=True)
        
        # Display table
        from rich.table import Table
        table = Table(title="Conversation History")
        table.add_column("ID")
        table.add_column("Start Time")
        table.add_column("Messages")
        table.add_column("Backend")
        table.add_column("Model")
        
        for conv in conversations:
            table.add_row(
                conv["id"],
                conv["start_time"],
                str(conv["messages"]),
                conv["backend"],
                conv["model"]
            )
            
        console.print(table)
        
    def _get_response(self, prompt: str) -> str:
        """Get response from the model.
        
        Args:
            prompt: User prompt
            
        Returns:
            str: Model response
        """
        llm = self.backend_manager.get_llm()
        
        # Build context from conversation history
        context = self._build_context()
        
        # Get response
        response = llm._call(
            f"{context}\nUser: {prompt}\nAssistant:"
        )
        
        return response.strip()
        
    def _build_context(self) -> str:
        """Build context from conversation history.
        
        Returns:
            str: Formatted context
        """
        context = []
        
        # Add system prompt if present
        system_prompt = self.current_conversation["metadata"].get("system_prompt")
        if system_prompt:
            context.append(f"System: {system_prompt}")
            
        # Add recent messages
        for msg in self.current_conversation["messages"][-5:]:  # Last 5 messages
            if msg["role"] == "system":
                continue
            role = "User" if msg["role"] == "user" else "Assistant"
            context.append(f"{role}: {msg['content']}")
            
        return "\n".join(context)
        
    def _handle_command(self, command: str):
        """Handle special commands.
        
        Args:
            command: Command string
        """
        cmd = command.lower()
        
        if cmd == '/help':
            self._show_help()
        elif cmd == '/history':
            self._show_conversation_history()
        elif cmd == '/clear':
            self.current_conversation["messages"] = []
            console.print("[green]Conversation history cleared")
        elif cmd == '/save':
            self._save_conversation()
            console.print("[green]Conversation saved")
        elif cmd == '/stats':
            self._show_stats()
        elif cmd.startswith('/switch'):
            parts = command.split()
            if len(parts) > 1:
                self._switch_model(parts[1])
        else:
            console.print("[yellow]Unknown command. Type /help for available commands")
            
    def _show_help(self):
        """Show help information."""
        help_text = """
Available Commands:
------------------
/help     - Show this help message
/history  - Show conversation history
/clear    - Clear conversation history
/save     - Save conversation
/stats    - Show performance statistics
/switch   - Switch model (e.g., /switch gpt-4)
/quit     - End conversation
"""
        console.print(Panel(help_text, title="Help"))
        
    def _show_conversation_history(self):
        """Show conversation history."""
        for msg in self.current_conversation["messages"]:
            if msg["role"] == "system":
                continue
                
            content = msg["content"]
            if msg["role"] == "user":
                console.print(f"\n[bold blue]You:[/bold blue]")
                console.print(content)
            else:
                console.print(f"\n[bold green]Assistant:[/bold green]")
                self._display_response(content)
                
    def _show_stats(self):
        """Show performance statistics."""
        stats = self.performance_monitor.get_summary()
        if not stats:
            console.print("[yellow]No statistics available")
            return
            
        console.print(Panel.fit(
            f"""Messages: {stats['total_inferences']}
Average Response Time: {stats['avg_latency']:.2f}s
P95 Response Time: {stats['p95_latency']:.2f}s
Memory Usage: {stats['avg_memory_mb']:.1f}MB""",
            title="Performance Statistics"
        ))
        
    def _switch_model(self, model_name: str):
        """Switch to a different model.
        
        Args:
            model_name: Name of model to switch to
        """
        try:
            self.backend_manager.add_model(
                self.backend_manager.current_backend,
                model_name,
                {}
            )
            self.current_conversation["metadata"]["model"] = model_name
            console.print(f"[green]Switched to model: {model_name}")
        except Exception as e:
            console.print(f"[red]Error switching model: {str(e)}")
            
    def _save_conversation(self):
        """Save conversation to disk."""
        filepath = self.history_dir / f"{self.current_conversation['id']}.json"
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_conversation, f, indent=2)
            console.print(f"[green]Conversation saved to {filepath}")
        except Exception as e:
            console.print(f"[red]Error saving conversation: {str(e)}")
            
    def _show_conversation_start(self):
        """Show conversation start message."""
        metadata = self.current_conversation["metadata"]
        console.print(Panel.fit(
            f"""Backend: {metadata['backend']}
Model: {metadata['model']}
Type /help for available commands""",
            title="Starting Conversation",
            border_style="blue"
        ))
        
    def _display_response(self, response: str):
        """Display formatted response.
        
        Args:
            response: Response text to display
        """
        # Check if response contains code blocks
        if "```" in response:
            # Split response into text and code blocks
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Text part
                    if part.strip():
                        console.print(Markdown(part))
                else:  # Code part
                    lang = part.split('\n')[0]
                    code = '\n'.join(part.split('\n')[1:])
                    console.print(Syntax(code, lang, theme="monokai"))
        else:
            # Regular markdown rendering
            console.print(Markdown(response))