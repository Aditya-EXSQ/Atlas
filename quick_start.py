#!/usr/bin/env python3
"""
Quick Start Guide for ChatBot
==============================
This script helps you get started with the chatbot.
"""


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    print_section("ChatBot - Quick Start Guide")

    print("Your chatbot is ready! Here's how to get started:\n")

    print("📋 OPTION 1: Using Ollama (Recommended for Beginners)")
    print("-" * 60)
    print("1. Install Ollama:")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("\n2. Pull a model:")
    print("   ollama pull mistral:7b-instruct")
    print("\n3. Start Ollama server (in one terminal):")
    print("   ollama serve")
    print("\n4. Install Python dependencies (in another terminal):")
    print("   pip install -r requirements.txt")
    print("\n5. (Optional) Verify installation:")
    print("   python verify_setup.py")
    print("\n6. Run the chatbot:")
    print("   python main.py")

    print("\n\n📋 OPTION 2: Using HuggingFace Transformers")
    print("-" * 60)
    print("1. Install PyTorch with CUDA 12:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("\n2. Install other dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Edit config/model_config.yaml:")
    print("   Change 'backend: \"ollama\"' to 'backend: \"transformers\"'")
    print("\n4. (Optional) Login to HuggingFace for gated models:")
    print("   pip install huggingface_hub")
    print("   huggingface-cli login")
    print("\n5. (Optional) Verify installation:")
    print("   python verify_setup.py")
    print("\n6. Run the chatbot:")
    print("   python main.py")

    print("\n\n💡 Tips:")
    print("-" * 60)
    print("• Check README.md for detailed documentation")
    print("• Use verify_setup.py to check if everything is installed")
    print("• Configuration is in config/model_config.yaml")
    print("• Type 'exit', 'quit', or 'bye' to end conversation")
    print("• Conversation history is saved in faiss_index/")

    print("\n\n🔧 Troubleshooting:")
    print("-" * 60)
    print("• CUDA out of memory? Enable 4-bit quantization in config")
    print("• Ollama connection error? Make sure 'ollama serve' is running")
    print("• FAISS GPU error? Install faiss-cpu instead of faiss-gpu")

    print("\n\n📚 Project Structure:")
    print("-" * 60)
    print("config/          - Configuration files")
    print("models/          - LLM and embedding handlers")
    print("memory/          - Conversation manager and RAG store")
    print("utils/           - Utility functions")
    print("main.py          - Main chatbot application")
    print("verify_setup.py  - Dependency verification script")

    print_section("Ready to Chat!")

    # Check if Ollama is running
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama is running! You can start with:")
            print("  python main.py\n")
        else:
            print("⚠ Start Ollama first:")
            print("  ollama serve\n")
    except:
        print("💡 To use Ollama backend, start it first:")
        print("   ollama serve")
        print("\n   Or switch to Transformers backend in config/model_config.yaml\n")


if __name__ == "__main__":
    main()
