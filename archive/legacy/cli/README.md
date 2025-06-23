# LLM Manager CLI

A command-line interface for managing LLM backends and models.

## Installation

The CLI is included in the main package. Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### List Available Models

List all available models across all backends:
```bash
python -m cli.llm_manager_cli list
```

Filter by backend:
```bash
python -m cli.llm_manager_cli list --backend llamafile
```

### Switch Backends

Switch to a different backend:
```bash
python -m cli.llm_manager_cli switch llamafile
```

Switch to a specific model:
```bash
python -m cli.llm_manager_cli switch llamafile --model llama-2-7b
```

### Set Up Models

Set up all Llamafile models:
```bash
python -m cli.llm_manager_cli setup --llamafile all
```

Set up a specific model:
```bash
python -m cli.llm_manager_cli setup --llamafile llama-2-7b
```

Show LM Studio setup instructions:
```bash
python -m cli.llm_manager_cli setup --lmstudio
```

### Check Status

Show current system status:
```bash
python -m cli.llm_manager_cli status
```

### Test Models

Test current model:
```bash
python -m cli.llm_manager_cli test --prompt "What is machine learning?"
```

Test specific backend:
```bash
python -m cli.llm_manager_cli test --backend lmstudio --prompt "Explain Python to me"
```

### Manage Configuration

Show current configuration:
```bash
python -m cli.llm_manager_cli config --show
```

## Environment Variables

The CLI respects the following environment variables:

- `OPENAI_API_KEY`: API key for OpenAI models
- `LMSTUDIO_URL`: URL for LM Studio API (default: http://localhost:1234/v1)
- `LLAMAFILE_MODELS_DIR`: Directory for storing Llamafile models

## Examples

1. Complete setup of local models:
```bash
# Set up Llamafile models
python -m cli.llm_manager_cli setup --llamafile all

# Get LM Studio instructions
python -m cli.llm_manager_cli setup --lmstudio

# Check status
python -m cli.llm_manager_cli status
```

2. Switch between backends:
```bash
# Use OpenAI
python -m cli.llm_manager_cli switch openai

# Use local model
python -m cli.llm_manager_cli switch llamafile --model llama-2-7b

# Test current model
python -m cli.llm_manager_cli test --prompt "Write a Python function to calculate Fibonacci numbers"
```

## Error Handling

The CLI provides clear error messages and status information. Common issues:

1. OpenAI API key not set:
```bash
export OPENAI_API_KEY=your_api_key
```

2. LM Studio not running:
```bash
# Start LM Studio and ensure server is running
python -m cli.llm_manager_cli status
```

3. Llamafile models not installed:
```bash
python -m cli.llm_manager_cli setup --llamafile all
```