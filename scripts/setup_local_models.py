"""Script to help users download and set up local models."""
import os
import sys
import argparse
import subprocess
import requests
from typing import Optional
from tqdm import tqdm
from config.llm_config import (
    DEFAULT_MODELS_DIR,
    LLAMAFILE_CONFIG,
    get_model_path
)

def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
        chunk_size: Size of chunks to download
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def setup_llamafile(model_name: str):
    """Download and set up a Llamafile model.
    
    Args:
        model_name: Name of the model to set up
    """
    if model_name not in LLAMAFILE_CONFIG["models"]:
        print(f"Error: Unknown model {model_name}")
        return False
        
    model_path = get_model_path(model_name)
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return True
        
    # URLs for different models
    model_urls = {
        "llama-2-7b": "https://huggingface.co/jartine/llama-2-7b-llamafile/resolve/main/llama-2-7b-chat.Q4_K_M.llamafile",
        "llama-2-13b": "https://huggingface.co/jartine/llama-2-13b-llamafile/resolve/main/llama-2-13b-chat.Q4_K_M.llamafile",
        "codellama-7b": "https://huggingface.co/jartine/codellama-7b-llamafile/resolve/main/codellama-7b.Q4_K_M.llamafile"
    }
    
    if model_name not in model_urls:
        print(f"Error: No download URL for model {model_name}")
        return False
        
    print(f"Downloading {model_name}...")
    try:
        download_file(model_urls[model_name], model_path)
        os.chmod(model_path, 0o755)  # Make executable
        print(f"Successfully downloaded and set up {model_name}")
        return True
    except Exception as e:
        print(f"Error downloading {model_name}: {str(e)}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False

def setup_lmstudio():
    """Help user set up LM Studio."""
    print("\nSetting up LM Studio:")
    print("1. Download LM Studio from https://lmstudio.ai/")
    print("2. Install and launch LM Studio")
    print("3. Download a model through LM Studio's interface")
    print("   Recommended models:")
    print("   - TheBloke/Llama-2-7B-Chat-GGUF")
    print("   - TheBloke/CodeLlama-7B-Instruct-GGUF")
    print("   - TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
    print("4. Start the local server in LM Studio:")
    print("   - Click 'Local Server' in the sidebar")
    print("   - Select your downloaded model")
    print("   - Click 'Start Server'")
    print("5. Set environment variables (optional):")
    print("   export LMSTUDIO_URL=http://localhost:1234/v1")
    
    return True

def verify_setup():
    """Verify the setup of local models."""
    print("\nVerifying setup...")
    
    # Check Llamafile models
    print("\nLlamafile models:")
    for model_name in LLAMAFILE_CONFIG["models"]:
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            print(f"✓ {model_name} is installed at {model_path}")
        else:
            print(f"✗ {model_name} is not installed")
            
    # Check LM Studio
    print("\nLM Studio:")
    try:
        response = requests.get(os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1"))
        if response.status_code == 200:
            print("✓ LM Studio server is running")
        else:
            print("✗ LM Studio server is not responding")
    except:
        print("✗ LM Studio server is not running")
        
    print("\nSetup verification complete.")

def main():
    parser = argparse.ArgumentParser(description="Set up local LLM models")
    parser.add_argument(
        "--model",
        choices=["all"] + list(LLAMAFILE_CONFIG["models"].keys()),
        default="all",
        help="Model to set up (default: all)"
    )
    parser.add_argument(
        "--setup-lmstudio",
        action="store_true",
        help="Show instructions for setting up LM Studio"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the setup of local models"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_setup()
        return
        
    if args.setup_lmstudio:
        setup_lmstudio()
        return
        
    # Create models directory if it doesn't exist
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    
    if args.model == "all":
        success = True
        for model_name in LLAMAFILE_CONFIG["models"]:
            if not setup_llamafile(model_name):
                success = False
        sys.exit(0 if success else 1)
    else:
        sys.exit(0 if setup_llamafile(args.model) else 1)

if __name__ == "__main__":
    main()