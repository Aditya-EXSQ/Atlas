#!/usr/bin/env python3
"""
Quick verification script to check if all dependencies are installed correctly.
Run this before using the chatbot to ensure everything is set up properly.
"""

import sys


def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    display_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name} - {e}")
        return False


def main():
    print("=" * 60)
    print("ChatBot Dependency Verification")
    print("=" * 60)
    
    all_ok = True
    
    print("\n[Core Dependencies]")
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("transformers", "Transformers")
    all_ok &= check_import("bitsandbytes", "BitsAndBytes")
    all_ok &= check_import("accelerate", "Accelerate")
    
    print("\n[Embedding & Search]")
    all_ok &= check_import("sentence_transformers", "Sentence Transformers")
    all_ok &= check_import("faiss", "FAISS")
    
    print("\n[Utilities]")
    all_ok &= check_import("yaml", "PyYAML")
    all_ok &= check_import("requests", "Requests")
    all_ok &= check_import("colorama", "Colorama")
    
    # Check CUDA availability
    print("\n[GPU Check]")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠ CUDA not available - will run on CPU (slower)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        all_ok = False
    
    # Check FAISS GPU support
    print("\n[FAISS GPU Check]")
    try:
        import faiss
        if hasattr(faiss, 'get_num_gpus'):
            num_gpus = faiss.get_num_gpus()
            if num_gpus > 0:
                print(f"✓ FAISS GPU support: {num_gpus} GPU(s)")
            else:
                print("⚠ FAISS CPU version (GPU version recommended)")
        else:
            print("⚠ FAISS CPU version (GPU version recommended)")
    except Exception as e:
        print(f"✗ Error checking FAISS: {e}")
    
    # Check Ollama connection (optional)
    print("\n[Optional: Ollama Check]")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ Ollama is running")
            if models:
                print(f"  Available models: {', '.join([m['name'] for m in models[:3]])}")
            else:
                print("  ⚠ No models installed. Run: ollama pull mistral:7b-instruct")
        else:
            print("⚠ Ollama is running but returned unexpected status")
    except Exception:
        print("⚠ Ollama not running (only needed if using Ollama backend)")
        print("  To start: ollama serve")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All core dependencies are installed correctly!")
        print("You can now run: python main.py")
    else:
        print("✗ Some dependencies are missing. Please install them:")
        print("pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
