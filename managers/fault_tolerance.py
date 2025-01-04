from typing import Dict, Any, Optional, List, Union, Callable
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import signal
import sys
from utils.logging_util import LoggerMixin

class FaultType(Enum):
    """Types of faults."""
    PROCESS_FAILURE = "process_failure"
    GPU_ERROR = "gpu_error"
    OOM = "out_of_memory"
    NETWORK = "network_failure"
    TIMEOUT = "timeout"

@dataclass
class FaultConfig:
    """Fault tolerance configuration."""
    max_retries: int = 3
    retry_delay: float = 5.0  # seconds
    timeout: float = 300.0  # seconds
    checkpoint_interval: int = 100  # iterations
    enable_process_recovery: bool = True
    enable_gpu_recovery: bool = True
    enable_network_recovery: bool = True

class FaultHandler(LoggerMixin):
    """Fault handling and recovery system."""
    
    def __init__(self,
                 config: FaultConfig):
        """Initialize handler.
        
        Args:
            config: Fault tolerance configuration
        """
        super().__init__()
        self.config = config
        
        # Initialize state
        self.active = False
        self.fault_queue = queue.Queue()
        self.recovery_thread = None
        
        # Initialize process monitoring
        self.processes: Dict[int, mp.Process] = {}
        self.process_status: Dict[int, bool] = {}
        
        # Initialize GPU monitoring
        if torch.cuda.is_available():
            self.gpu_status = {
                i: True
                for i in range(torch.cuda.device_count())
            }
            
        # Set up signal handlers
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Set up signal handlers."""
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)
        
    def _handle_termination(self, signum, frame):
        """Handle termination signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.log_event(
            "termination_signal",
            {"signal": signum}
        )
        self.cleanup()
        sys.exit(0)
        
    def start(self):
        """Start fault handler."""
        if self.active:
            return
            
        self.active = True
        
        # Start recovery thread
        self.recovery_thread = threading.Thread(
            target=self._recovery_loop
        )
        self.recovery_thread.start()
        
        self.log_event(
            "fault_handler_started",
            {"config": self.config.__dict__}
        )
        
    def stop(self):
        """Stop fault handler."""
        if not self.active:
            return
            
        self.active = False
        
        if self.recovery_thread:
            self.recovery_thread.join()
            
        self.log_event("fault_handler_stopped")
        
    def register_process(self,
                        process: mp.Process,
                        rank: int):
        """Register process for monitoring.
        
        Args:
            process: Multiprocessing process
            rank: Process rank
        """
        self.processes[rank] = process
        self.process_status[rank] = True
        
        self.log_event(
            "process_registered",
            {"rank": rank, "pid": process.pid}
        )
        
    def monitor_gpu(self,
                   device: int,
                   threshold_mb: int = 100):
        """Monitor GPU memory and errors.
        
        Args:
            device: GPU device ID
            threshold_mb: Memory threshold in MB
        """
        try:
            # Check memory
            memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
            if memory > threshold_mb:
                self.report_fault(
                    FaultType.OOM,
                    {
                        "device": device,
                        "memory_mb": memory,
                        "threshold_mb": threshold_mb
                    }
                )
                
            # Check errors
            if torch.cuda.device_count() > device:
                if not torch.cuda.device(device).is_available():
                    self.report_fault(
                        FaultType.GPU_ERROR,
                        {"device": device}
                    )
                    
        except Exception as e:
            self.log_error(e, {"device": device})
            
    def report_fault(self,
                    fault_type: FaultType,
                    details: Optional[Dict[str, Any]] = None):
        """Report system fault.
        
        Args:
            fault_type: Type of fault
            details: Fault details
        """
        fault = {
            "type": fault_type.value,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.fault_queue.put(fault)
        
        self.log_event(
            "fault_reported",
            fault
        )
        
    def _recovery_loop(self):
        """Fault recovery loop."""
        while self.active:
            try:
                # Get fault
                try:
                    fault = self.fault_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Handle fault
                fault_type = FaultType(fault["type"])
                
                if fault_type == FaultType.PROCESS_FAILURE:
                    self._handle_process_failure(fault)
                    
                elif fault_type == FaultType.GPU_ERROR:
                    self._handle_gpu_error(fault)
                    
                elif fault_type == FaultType.OOM:
                    self._handle_oom(fault)
                    
                elif fault_type == FaultType.NETWORK:
                    self._handle_network_failure(fault)
                    
                elif fault_type == FaultType.TIMEOUT:
                    self._handle_timeout(fault)
                    
            except Exception as e:
                self.log_error(e)
                time.sleep(1.0)
                
    def _handle_process_failure(self, fault: Dict[str, Any]):
        """Handle process failure.
        
        Args:
            fault: Fault information
        """
        if not self.config.enable_process_recovery:
            return
            
        try:
            rank = fault["details"]["rank"]
            process = self.processes.get(rank)
            
            if not process:
                return
                
            # Check retries
            retries = fault["details"].get("retries", 0)
            if retries >= self.config.max_retries:
                self.log_event(
                    "max_retries_exceeded",
                    {"rank": rank}
                )
                return
                
            # Terminate process
            if process.is_alive():
                process.terminate()
                process.join()
                
            # Start new process
            new_process = mp.Process(
                target=fault["details"]["target"],
                args=fault["details"]["args"]
            )
            new_process.start()
            
            # Update state
            self.processes[rank] = new_process
            self.process_status[rank] = True
            
            self.log_event(
                "process_recovered",
                {
                    "rank": rank,
                    "new_pid": new_process.pid
                }
            )
            
        except Exception as e:
            self.log_error(e, fault)
            
    def _handle_gpu_error(self, fault: Dict[str, Any]):
        """Handle GPU error.
        
        Args:
            fault: Fault information
        """
        if not self.config.enable_gpu_recovery:
            return
            
        try:
            device = fault["details"]["device"]
            
            # Reset device
            torch.cuda.device(device).empty_cache()
            torch.cuda.device(device).reset()
            
            # Update status
            self.gpu_status[device] = True
            
            self.log_event(
                "gpu_recovered",
                {"device": device}
            )
            
        except Exception as e:
            self.log_error(e, fault)
            
    def _handle_oom(self, fault: Dict[str, Any]):
        """Handle out of memory error.
        
        Args:
            fault: Fault information
        """
        try:
            device = fault["details"]["device"]
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Reduce batch size if possible
            if "batch_size" in fault["details"]:
                new_batch_size = fault["details"]["batch_size"] // 2
                fault["details"]["new_batch_size"] = new_batch_size
                
            self.log_event(
                "oom_handled",
                {
                    "device": device,
                    "details": fault["details"]
                }
            )
            
        except Exception as e:
            self.log_error(e, fault)
            
    def _handle_network_failure(self, fault: Dict[str, Any]):
        """Handle network failure.
        
        Args:
            fault: Fault information
        """
        if not self.config.enable_network_recovery:
            return
            
        try:
            # Reinitialize process group
            if dist.is_initialized():
                dist.destroy_process_group()
                
            # Wait before retry
            time.sleep(self.config.retry_delay)
            
            # Initialize new process group
            dist.init_process_group(
                backend=fault["details"].get("backend", "nccl"),
                init_method=fault["details"].get("init_method"),
                world_size=fault["details"].get("world_size"),
                rank=fault["details"].get("rank")
            )
            
            self.log_event(
                "network_recovered",
                fault["details"]
            )
            
        except Exception as e:
            self.log_error(e, fault)
            
    def _handle_timeout(self, fault: Dict[str, Any]):
        """Handle timeout.
        
        Args:
            fault: Fault information
        """
        try:
            # Log timeout
            self.log_event(
                "timeout_occurred",
                fault["details"]
            )
            
            # Implement timeout recovery logic
            # This could involve:
            # - Restarting processes
            # - Reducing batch size
            # - Adjusting learning rate
            # - etc.
            
        except Exception as e:
            self.log_error(e, fault)
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status.
        
        Returns:
            Dict[str, Any]: Health information
        """
        return {
            "processes": {
                rank: {
                    "alive": process.is_alive(),
                    "pid": process.pid,
                    "status": self.process_status[rank]
                }
                for rank, process in self.processes.items()
            },
            "gpus": {
                device: {
                    "available": torch.cuda.device(device).is_available(),
                    "memory": {
                        "allocated": torch.cuda.memory_allocated(device),
                        "cached": torch.cuda.memory_reserved(device)
                    },
                    "status": self.gpu_status.get(device, False)
                }
                for device in range(torch.cuda.device_count())
            } if torch.cuda.is_available() else {},
            "network": {
                "initialized": dist.is_initialized(),
                "world_size": dist.get_world_size() if dist.is_initialized() else 0
            }
        }
        
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop handler
            self.stop()
            
            # Terminate processes
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
                    process.join()
                    
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Clean up distributed
            if dist.is_initialized():
                dist.destroy_process_group()
                
        except Exception as e:
            self.log_error(e)
            
    def __del__(self):
        """Clean up on deletion."""
        self.cleanup()