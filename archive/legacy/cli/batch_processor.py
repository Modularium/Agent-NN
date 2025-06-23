"""Batch processing for LLM tasks."""
import os
import csv
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.console import Console
from rich.table import Table

from llm_models.llm_backend import LLMBackendManager, LLMBackendType
from .performance_monitor import PerformanceMonitor

console = Console()

class BatchProcessor:
    def __init__(self, output_dir: str = "output/batch"):
        """Initialize batch processor.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.backend_manager = LLMBackendManager()
        self.performance_monitor = PerformanceMonitor()
        
    async def process_file(self,
                         input_file: str,
                         backend_type: Optional[LLMBackendType] = None,
                         model_name: Optional[str] = None,
                         batch_size: int = 10,
                         max_retries: int = 3,
                         output_format: str = "json"):
        """Process a file containing prompts in batch.
        
        Args:
            input_file: Path to input file (CSV or JSON)
            backend_type: Optional backend to use
            model_name: Optional specific model to use
            batch_size: Number of prompts to process in parallel
            max_retries: Maximum number of retries for failed requests
            output_format: Output format (json or csv)
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
            
        # Load prompts
        prompts = self._load_prompts(input_file)
        if not prompts:
            console.print("[red]No prompts found in input file")
            return
            
        # Start performance monitoring
        self.performance_monitor.start_monitoring(
            model_name or "default",
            self.backend_manager.current_backend.value
        )
        
        # Process prompts in batches
        results = []
        failed = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(
                "Processing prompts...",
                total=len(prompts)
            )
            
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                batch_results = await self._process_batch(
                    batch,
                    max_retries,
                    progress,
                    task
                )
                
                results.extend(batch_results["success"])
                failed.extend(batch_results["failed"])
                
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{timestamp}_batch_results"
        
        if output_format == "json":
            self._save_json(results, failed, output_base)
        else:
            self._save_csv(results, failed, output_base)
            
        # Generate performance report
        self.performance_monitor.generate_report(
            str(self.output_dir / f"{output_base}_performance.txt")
        )
        self.performance_monitor.save_session()
        
        # Show summary
        self._show_summary(results, failed)
        
    async def _process_batch(self,
                           batch: List[Dict[str, Any]],
                           max_retries: int,
                           progress: Progress,
                           task) -> Dict[str, List]:
        """Process a batch of prompts.
        
        Args:
            batch: List of prompts to process
            max_retries: Maximum number of retries
            progress: Progress bar
            task: Progress task
            
        Returns:
            Dict containing successful and failed results
        """
        llm = self.backend_manager.get_llm()
        tasks = []
        
        for prompt in batch:
            tasks.append(self._process_prompt(
                prompt,
                llm,
                max_retries
            ))
            
        results = await asyncio.gather(*tasks)
        
        success = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        progress.update(task, advance=len(batch))
        return {"success": success, "failed": failed}
        
    async def _process_prompt(self,
                            prompt: Dict[str, Any],
                            llm,
                            max_retries: int) -> Dict[str, Any]:
        """Process a single prompt with retries.
        
        Args:
            prompt: Prompt data
            llm: LLM instance
            max_retries: Maximum number of retries
            
        Returns:
            Dict containing result or error
        """
        retries = 0
        while retries < max_retries:
            try:
                start_time = asyncio.get_event_loop().time()
                response = await llm._acall(prompt["text"])
                end_time = asyncio.get_event_loop().time()
                
                self.performance_monitor.log_inference(
                    prompt["text"],
                    response,
                    start_time,
                    end_time,
                    prompt.get("metadata")
                )
                
                return {
                    "prompt": prompt,
                    "response": response,
                    "duration": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                retries += 1
                if retries == max_retries:
                    return {
                        "prompt": prompt,
                        "error": str(e),
                        "attempts": retries,
                        "timestamp": datetime.now().isoformat()
                    }
                await asyncio.sleep(1)  # Back off before retry
                
    def _load_prompts(self, input_file: str) -> List[Dict[str, Any]]:
        """Load prompts from file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of prompt dictionaries
        """
        ext = os.path.splitext(input_file)[1].lower()
        
        try:
            if ext == '.json':
                with open(input_file, 'r') as f:
                    data = json.load(f)
                    return [{"text": p} if isinstance(p, str) else p 
                           for p in data]
                           
            elif ext == '.csv':
                prompts = []
                with open(input_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "prompt" in row:
                            prompts.append({
                                "text": row["prompt"],
                                "metadata": {k: v for k, v in row.items() 
                                          if k != "prompt"}
                            })
                return prompts
                
            else:
                console.print(f"[red]Unsupported file format: {ext}")
                return []
                
        except Exception as e:
            console.print(f"[red]Error loading prompts: {str(e)}")
            return []
            
    def _save_json(self,
                  results: List[Dict[str, Any]],
                  failed: List[Dict[str, Any]],
                  base_name: str):
        """Save results in JSON format.
        
        Args:
            results: Successful results
            failed: Failed attempts
            base_name: Base name for output files
        """
        if results:
            with open(self.output_dir / f"{base_name}_success.json", 'w') as f:
                json.dump(results, f, indent=2)
                
        if failed:
            with open(self.output_dir / f"{base_name}_failed.json", 'w') as f:
                json.dump(failed, f, indent=2)
                
    def _save_csv(self,
                 results: List[Dict[str, Any]],
                 failed: List[Dict[str, Any]],
                 base_name: str):
        """Save results in CSV format.
        
        Args:
            results: Successful results
            failed: Failed attempts
            base_name: Base name for output files
        """
        if results:
            with open(self.output_dir / f"{base_name}_success.csv", 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "prompt", "response", "duration", "timestamp"
                ])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        "prompt": r["prompt"]["text"],
                        "response": r["response"],
                        "duration": r["duration"],
                        "timestamp": r["timestamp"]
                    })
                    
        if failed:
            with open(self.output_dir / f"{base_name}_failed.csv", 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "prompt", "error", "attempts", "timestamp"
                ])
                writer.writeheader()
                for r in failed:
                    writer.writerow({
                        "prompt": r["prompt"]["text"],
                        "error": r["error"],
                        "attempts": r["attempts"],
                        "timestamp": r["timestamp"]
                    })
                    
    def _show_summary(self,
                     results: List[Dict[str, Any]],
                     failed: List[Dict[str, Any]]):
        """Show processing summary.
        
        Args:
            results: Successful results
            failed: Failed attempts
        """
        table = Table(title="Batch Processing Summary")
        table.add_column("Metric")
        table.add_column("Value")
        
        total = len(results) + len(failed)
        success_rate = len(results) / total * 100 if total > 0 else 0
        
        table.add_row("Total Prompts", str(total))
        table.add_row("Successful", str(len(results)))
        table.add_row("Failed", str(len(failed)))
        table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        if results:
            durations = [r["duration"] for r in results]
            avg_duration = sum(durations) / len(durations)
            table.add_row("Average Duration", f"{avg_duration:.2f}s")
            
        console.print(table)