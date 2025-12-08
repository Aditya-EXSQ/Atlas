#!/usr/bin/env python3
"""
Test script for torch context-based model pipelines.

This demonstrates the three Transformers-based backends:
1. transformers - Standard implementation
2. transformers_no_grad - Uses torch.no_grad() during generation
3. transformers_inference_mode - Uses torch.inference_mode() during generation

All three should produce identical streaming output, but may have different
performance characteristics that can be benchmarked.
"""

import time
from models.llm_handler import create_llm


def test_pipeline(backend_name: str, config: dict):
    """
    Test a single pipeline backend.

    Args:
        backend_name: Name of the backend for display
        config: Configuration dictionary
    """
    print("=" * 70)
    print(f"Testing: {backend_name}")
    print("=" * 70)

    # Create LLM instance
    print(f"\n1. Creating {backend_name} instance...")
    llm = create_llm(config)

    # Test generation
    print("\n2. Testing streaming generation...")
    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate fibonacci numbers.",
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
    print(
        f"Tokens per second (approx): {len(full_response.split()) / (end_time - start_time):.2f}"
    )
    print(f"\n✓ {backend_name} test completed!\n")

    return end_time - start_time


def test_all_pipelines():
    """Test all three Transformers-based pipelines."""

    # Base configuration (shared across all backends)
    base_config = {
        "model_name": "microsoft/phi-2",  # Small model for quick testing
        "device": "cuda",
        "load_in_4bit": True,
        "temperature": 0.7,
        "max_new_tokens": 150,
    }

    results = {}

    # Test 1: Standard Transformers
    config_1 = {"backend": "transformers", "transformers": base_config.copy()}
    results["transformers"] = test_pipeline("Standard Transformers", config_1)

    # Test 2: Transformers with torch.no_grad()
    config_2 = {
        "backend": "transformers_no_grad",
        "transformers_no_grad": base_config.copy(),
    }
    results["transformers_no_grad"] = test_pipeline(
        "Transformers + torch.no_grad()", config_2
    )

    # Test 3: Transformers with torch.inference_mode()
    config_3 = {
        "backend": "transformers_inference_mode",
        "transformers_inference_mode": base_config.copy(),
    }
    results["transformers_inference_mode"] = test_pipeline(
        "Transformers + torch.inference_mode()", config_3
    )

    # Summary
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print("\nLatency Comparison:")
    for backend, latency in results.items():
        print(f"  {backend:30s}: {latency:.2f}s")

    fastest = min(results.items(), key=lambda x: x[1])
    print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1]:.2f}s)")
    print("\n" + "=" * 70)


def test_single_backend(backend: str):
    """
    Test a single backend for quick verification.

    Args:
        backend: One of 'transformers', 'transformers_no_grad', 'transformers_inference_mode'
    """
    config = {
        "backend": backend,
        backend: {
            "model_name": "microsoft/phi-2",
            "device": "cuda",
            "load_in_4bit": True,
            "temperature": 0.7,
            "max_new_tokens": 150,
        },
    }

    test_pipeline(backend.replace("_", " ").title(), config)


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("Torch Context Pipelines Test Suite")
    print("=" * 70)

    if len(sys.argv) > 1:
        # Test specific backend
        backend = sys.argv[1]
        if backend not in [
            "transformers",
            "transformers_no_grad",
            "transformers_inference_mode",
        ]:
            print(f"\n❌ Invalid backend: {backend}")
            print(
                "Valid options: transformers, transformers_no_grad, transformers_inference_mode"
            )
            sys.exit(1)

        print(f"\nTesting single backend: {backend}\n")
        test_single_backend(backend)
    else:
        # Test all backends
        print("\nTesting all three backends for comparison...\n")
        try:
            test_all_pipelines()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
