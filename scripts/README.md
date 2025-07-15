# Utility Scripts

This directory contains utility scripts for setting up and managing the system.
New helper scripts simplify installation and troubleshooting:

- `install_dependencies.sh` installs system packages for different presets.
- `install_dev_env.sh` wraps the above for a full developer setup.
- `repair_env.sh` checks the environment and installs missing tools.
- `check_integrity.sh` verifies that important files exist.

## Setup Local Models

The `setup_local_models.py` script helps you download and set up local LLM models. It supports both Llamafile models and provides guidance for LM Studio setup.

### Usage

1. Set up all Llamafile models:
```bash
python scripts/setup_local_models.py --model all
```

2. Set up a specific model:
```bash
python scripts/setup_local_models.py --model llama-2-7b
```

3. Get LM Studio setup instructions:
```bash
python scripts/setup_local_models.py --setup-lmstudio
```

4. Verify your setup:
```bash
python scripts/setup_local_models.py --verify
```

### Supported Models

#### Llamafile Models
- llama-2-7b: Llama 2 7B base model
- llama-2-13b: Llama 2 13B base model
- codellama-7b: CodeLlama 7B for technical tasks

#### LM Studio
LM Studio supports various models from Hugging Face. Recommended models:
- TheBloke/Llama-2-7B-Chat-GGUF
- TheBloke/CodeLlama-7B-Instruct-GGUF
- TheBloke/Mistral-7B-Instruct-v0.2-GGUF

### Environment Variables

- `LLAMAFILE_MODELS_DIR`: Directory for storing Llamafile models (default: `models/`)
- `LMSTUDIO_URL`: URL for LM Studio API (default: `http://localhost:1234/v1`)