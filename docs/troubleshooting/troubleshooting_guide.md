# Troubleshooting Guide

This guide helps diagnose and resolve common issues in the Smolit LLM-NN system.

## Quick Reference

### Common Issues

1. Performance Issues
   - [Slow Response Times](#slow-response-times)
   - [High Resource Usage](#high-resource-usage)
   - [GPU Problems](#gpu-problems)

2. System Issues
   - [Startup Failures](#startup-failures)
   - [Crashes](#crashes)
   - [Memory Leaks](#memory-leaks)

3. Model Issues
   - [Loading Failures](#model-loading-failures)
   - [Inference Errors](#inference-errors)
   - [GPU Memory Issues](#gpu-memory-issues)

4. API Issues
   - [Connection Problems](#connection-problems)
   - [Authentication Errors](#authentication-errors)
   - [Rate Limiting](#rate-limiting)

## Diagnostic Tools

### System Health Check

```bash
# Check system status
python -m smolit.diagnostics check

# Get detailed metrics
python -m smolit.diagnostics metrics

# Test GPU availability
python -m smolit.diagnostics gpu
```

### Log Analysis

```bash
# View system logs
python -m smolit.logs view

# Search for errors
python -m smolit.logs search "error"

# Export logs
python -m smolit.logs export --start "2024-01-01" --end "2024-01-02"
```

### Performance Testing

```bash
# Run performance tests
python -m smolit.test performance

# Run stress tests
python -m smolit.test stress

# Test GPU performance
python -m smolit.test gpu
```

## Common Issues

### Slow Response Times

#### Symptoms
- API requests take longer than usual
- Model inference is slow
- System feels unresponsive

#### Diagnosis
1. Check system metrics:
   ```bash
   python -m smolit.diagnostics metrics
   ```

2. Monitor resource usage:
   ```bash
   python -m smolit.monitor resources
   ```

3. Check GPU utilization:
   ```bash
   nvidia-smi -l 1
   ```

#### Solutions
1. Resource Optimization:
   ```python
   # Update system configuration
   await system_manager.update_config({
       "max_concurrent_tasks": 5,
       "cache_size": 512
   })
   ```

2. Cache Tuning:
   ```python
   # Clear cache
   cache_manager.clear()
   
   # Adjust cache size
   cache_manager = CacheManager(max_size=2048)  # 2GB
   ```

3. GPU Optimization:
   ```python
   # Enable GPU memory optimization
   torch.cuda.empty_cache()
   torch.backends.cudnn.benchmark = True
   ```

### High Resource Usage

#### Symptoms
- High CPU/Memory usage
- GPU memory exhaustion
- System slowdown

#### Diagnosis
1. Check process usage:
   ```bash
   top -p $(pgrep -d',' -f smolit)
   ```

2. Monitor GPU:
   ```bash
   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
   ```

3. Check memory leaks:
   ```bash
   python -m memory_profiler smolit_app.py
   ```

#### Solutions
1. Resource Limits:
   ```python
   # Set resource limits
   import resource
   resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
   ```

2. GPU Memory Management:
   ```python
   # Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Use mixed precision
   from torch.cuda.amp import autocast
   with autocast():
       outputs = model(inputs)
   ```

3. Process Management:
   ```python
   # Implement graceful shutdown
   import signal
   
   def shutdown_handler(signum, frame):
       cleanup()
       sys.exit(0)
       
   signal.signal(signal.SIGTERM, shutdown_handler)
   ```

### GPU Problems

#### Symptoms
- GPU not detected
- CUDA errors
- Out of memory errors

#### Diagnosis
1. Check GPU status:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Monitor memory:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
   ```

3. Check CUDA version:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

#### Solutions
1. CUDA Setup:
   ```bash
   # Install CUDA toolkit
   sudo apt install nvidia-cuda-toolkit
   
   # Set environment variables
   export CUDA_HOME=/usr/local/cuda
   export PATH=$PATH:$CUDA_HOME/bin
   ```

2. PyTorch Configuration:
   ```python
   # Configure PyTorch
   torch.backends.cudnn.benchmark = True
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

3. Memory Management:
   ```python
   # Implement memory efficient loading
   def load_model(path: str):
       with torch.cuda.device(0):
           torch.cuda.empty_cache()
           model = torch.load(path, map_location="cuda:0")
       return model
   ```

## System Recovery

### Emergency Shutdown

```bash
# Graceful shutdown
python -m smolit.control shutdown

# Force shutdown
python -m smolit.control shutdown --force
```

### Backup Recovery

```bash
# List backups
python -m smolit.backup list

# Restore from backup
python -m smolit.backup restore <backup_id>
```

### Cache Reset

```bash
# Clear all caches
python -m smolit.cache clear

# Clear specific cache
python -m smolit.cache clear --type model
```

## Performance Optimization

### Model Optimization

```python
# Enable FP16 inference
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)

# Enable model parallelism
model = torch.nn.DataParallel(model)

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

### Memory Optimization

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use memory efficient attention
from transformers import GPTNeoForCausalLM
model = GPTNeoForCausalLM.from_pretrained(
    "model_name",
    use_cache=False,
    gradient_checkpointing=True
)

# Implement memory monitoring
def monitor_memory():
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.get_device_properties(i).total_memory
        allocated = torch.cuda.memory_allocated(i)
        gpu_memory.append({
            "device": i,
            "total": memory,
            "allocated": allocated,
            "free": memory - allocated
        })
    return gpu_memory
```

### Cache Optimization

```python
# Configure cache
cache_config = {
    "policy": "lru",
    "max_size": 1024,  # MB
    "ttl": 3600,  # seconds
    "cleanup_interval": 300  # seconds
}

# Monitor cache performance
def monitor_cache():
    stats = cache_manager.get_stats()
    print(f"Hit rate: {stats['hit_rate']:.2f}")
    print(f"Miss rate: {stats['miss_rate']:.2f}")
    print(f"Evictions: {stats['evictions']}")
```

## Monitoring

### Resource Monitoring

```python
# Monitor system resources
def monitor_resources():
    import psutil
    import GPUtil
    
    # CPU & Memory
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # GPU
    gpus = GPUtil.getGPUs()
    gpu_stats = [{
        "id": gpu.id,
        "load": gpu.load,
        "memory": {
            "total": gpu.memoryTotal,
            "used": gpu.memoryUsed,
            "free": gpu.memoryFree
        },
        "temperature": gpu.temperature
    } for gpu in gpus]
    
    return {
        "cpu": cpu_percent,
        "memory": memory._asdict(),
        "gpu": gpu_stats
    }
```

### Performance Monitoring

```python
# Monitor performance metrics
def monitor_performance():
    metrics = {
        "latency": [],
        "throughput": [],
        "error_rate": [],
        "gpu_utilization": []
    }
    
    def update_metrics(result):
        metrics["latency"].append(result["latency"])
        metrics["throughput"].append(result["throughput"])
        metrics["error_rate"].append(result["error_rate"])
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                util = torch.cuda.utilization(i)
                metrics["gpu_utilization"].append(util)
                
    return metrics
```

### Alert Configuration

```python
# Configure alerts
alerts_config = {
    "cpu_threshold": 80,  # %
    "memory_threshold": 85,  # %
    "gpu_memory_threshold": 90,  # %
    "latency_threshold": 1000,  # ms
    "error_rate_threshold": 0.01  # 1%
}

# Alert handler
def handle_alert(alert):
    print(f"Alert: {alert['type']}")
    print(f"Value: {alert['value']}")
    print(f"Threshold: {alert['threshold']}")
    print(f"Timestamp: {alert['timestamp']}")
```

## Best Practices

1. Regular Maintenance:
   - Monitor system metrics
   - Clear unused caches
   - Update model versions
   - Review error logs

2. Resource Management:
   - Set appropriate limits
   - Monitor usage patterns
   - Implement auto-scaling
   - Use resource pools

3. Error Handling:
   - Implement retries
   - Log errors properly
   - Set up alerts
   - Have fallback options

4. Performance Optimization:
   - Use caching effectively
   - Optimize model loading
   - Implement batching
   - Monitor and tune

## Support

For additional support:
1. Check documentation
2. Review issue tracker
3. Contact support team
4. Join community forum