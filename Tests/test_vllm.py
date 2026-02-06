#!/usr/bin/env python3
"""
Test script for vLLM OpenAI API implementation.

This demonstrates how to:
1. Start a vLLM server
2. Create a vLLM client
3. Generate streaming responses
4. Stop the server
"""

import time

from models.llm_handler import create_llm, vLLM


def test_vllm_basic():
    """Basic test of vLLM with OpenAI API."""

    # Configuration for vLLM
    config = {
        "backend": "vllm",
        "vllm": {
            "model_name": "microsoft/phi-2",
            "port": 8000,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,
            "temperature": 0.5,
            "max_tokens": 200,
        },
    }

    print("=" * 70)
    print("vLLM Test - OpenAI API Approach")
    print("=" * 70)

    # Create LLM instance
    print("\n1. Creating vLLM instance...")
    llm = create_llm(config)

    # Note: Server must be started separately before running this test
    # See instructions below

    # Test generation
    print("\n2. Testing generation...")
    messages = [
        {
            "role": "user",
            "content": "Generate a python code that accepts a list of numbers and returns the sum.",
        }
    ]

    print("\nPrompt:", messages[0]["content"])
    print("\nResponse (streaming):")
    print("-" * 70)

    start_time = time.time()
    full_response = ""

    for token in llm.generate(messages):
        print(token, end="", flush=True)
        full_response += token

    end_time = time.time()

    print()
    print("-" * 70)
    print(f"\nLatency: {end_time - start_time:.2f} seconds")
    print(f"Response length: {len(full_response)} characters")
    print("\n✓ Test completed successfully!")


def demo_server_management():
    """
    Demonstrates how to start and stop vLLM server programmatically.

    WARNING: This will actually start a vLLM server process!
    """

    config = {
        "model_name": "microsoft/phi-2",
        "port": 8000,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
    }

    print("=" * 70)
    print("Server Management Demo")
    print("=" * 70)

    # Create vLLM instance (will warn that server is not running)
    print("\n1. Creating vLLM instance (server not started yet)...")
    llm = vLLM(config)

    # Start server
    print("\n2. Starting vLLM server...")
    llm.start_server()

    # Now you can use it
    print("\n3. Testing generation...")
    messages = [{"role": "user", "content": "Write a hello world program in Python."}]

    print("\nResponse:")
    for token in llm.generate(messages):
        print(token, end="", flush=True)
    print()

    # Stop server when done
    print("\n4. Stopping server...")
    llm.stop_server()

    print("\n✓ Demo completed!")


def print_usage_instructions():
    """Print instructions for using vLLM."""

    print("\n" + "=" * 70)
    print("How to Use vLLM with OpenAI API")
    print("=" * 70)

    print("""
OPTION 1: Start server manually (RECOMMENDED)
----------------------------------------------
1. Open a terminal and activate your conda environment:
   $ conda activate chatbot

2. Start the vLLM server:
   $ vllm serve microsoft/phi-2 --port 8000 --trust-remote-code

3. Run your script that uses the vLLM class:
   $ python test_vllm.py

OPTION 2: Start server programmatically
----------------------------------------
Use the start_server() and stop_server() methods:

```python
from models.llm_handler import vLLM

config = {
    "model_name": "microsoft/phi-2",
    "port": 8000,
    "trust_remote_code": True,
}

llm = vLLM(config)
llm.start_server()  # Starts server in background

# Use the model
messages = [{"role": "user", "content": "Hello!"}]
for token in llm.generate(messages):
    print(token, end="")

llm.stop_server()  # Clean up
```

OPTION 3: Standalone server management
---------------------------------------
```python
from models.llm_handler import vLLM

config = {"model_name": "microsoft/phi-2", "port": 8000}
process = vLLM.start_server_standalone(config)

# ... your code ...

process.terminate()  # Stop when done
```

Configuration Options:
----------------------
- model_name: Model ID or path (required)
- port: Server port (default: 8000)
- base_url: Custom API URL (default: http://localhost:{port}/v1)
- trust_remote_code: Trust remote code (default: True)
- dtype: Data type ("auto", "float16", "bfloat16", etc.)
- quantization: Quantization method ("awq", "gptq", "squeezellm")
- tensor_parallel_size: Number of GPUs for parallelism
- gpu_memory_utilization: GPU memory fraction (0.0-1.0)
- max_model_len: Maximum context length
- temperature: Sampling temperature
- max_tokens: Maximum tokens to generate

For more options, see the vLLM documentation:
https://docs.vllm.ai/en/latest/
    """)


if __name__ == "__main__":
    print_usage_instructions()

    print("\n" + "=" * 70)
    print("Running basic test...")
    print("NOTE: Make sure vLLM server is running first!")
    print("=" * 70)

    try:
        test_vllm_basic()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nMake sure the vLLM server is running:")
        print("$ conda activate chatbot")
        print("$ vllm serve microsoft/phi-2 --port 8000 --trust-remote-code")
