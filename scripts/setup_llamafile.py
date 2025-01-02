#!/usr/bin/env python3
"""Script to download and set up GGUF models."""

import os
import sys
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
from config.llm_config import LLAMAFILE_CONFIG, get_model_path

GGUF_RELEASES = {
    "llama-2-7b": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "llama-2-13b": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "codellama-7b": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
}

def download_file(url: str, dest_path: str):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def setup_llamafile(model_name: str = None):
    """Download and set up a specific Llamafile model or all models if none specified."""
    models_dir = Path(LLAMAFILE_CONFIG["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_setup = [model_name] if model_name else GGUF_RELEASES.keys()
    
    for model in models_to_setup:
        if model not in LLAMAFILE_CONFIG["models"]:
            print(f"Unknown model: {model}")
            continue
            
        model_path = get_model_path(model)
        if os.path.exists(model_path):
            print(f"Model {model} already exists at {model_path}")
            continue
            
        print(f"Downloading {model}...")
        download_file(GGUF_RELEASES[model], model_path)
        print(f"Model {model} downloaded and set up at {model_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        setup_llamafile(sys.argv[1])
    else:
        setup_llamafile()