"""
Quantization Library Availability Detection and Configuration.

This module checks for available quantization libraries at runtime and provides
a centralized location for managing quantization-related imports and configuration.

Usage:
    from models.quantization_config import AUTOAWQ_AVAILABLE, AutoAWQForCausalLM
"""

import importlib.util
from typing import Any, Optional

# =============================================================================
# AWQ Quantization
# =============================================================================

# AWQ Quantization (via transformers built-in support)
AWQ_AVAILABLE = importlib.util.find_spec("transformers.quantizers.awq") is not None
if AWQ_AVAILABLE:
    from transformers import AwqConfig
else:
    AwqConfig: Optional[Any] = None  # type: ignore

# AutoAWQ (direct AWQ loading library, bypasses transformers)
AUTOAWQ_AVAILABLE = importlib.util.find_spec("awq") is not None
if AUTOAWQ_AVAILABLE:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        AUTOAWQ_AVAILABLE = False
        AutoAWQForCausalLM: Optional[Any] = None  # type: ignore
else:
    AutoAWQForCausalLM: Optional[Any] = None  # type: ignore


# =============================================================================
# Future Quantization Methods (TODO)
# =============================================================================

# GPTQ Quantization
# GPTQ_AVAILABLE = importlib.util.find_spec("auto_gptq") is not None
# if GPTQ_AVAILABLE:
#     from auto_gptq import AutoGPTQForCausalLM
# else:
#     AutoGPTQForCausalLM = None

# GGUF Format (via llama.cpp)
# GGUF_AVAILABLE = importlib.util.find_spec("llama_cpp") is not None
# if GGUF_AVAILABLE:
#     from llama_cpp import Llama
# else:
#     Llama = None

# FP8/INT8 Quantization
# Check for specific acceleration libraries as needed


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_quantization_methods() -> list[str]:
    """
    Get a list of available quantization methods.

    Returns:
        List of quantization method names that are available
    """
    available = []

    if AWQ_AVAILABLE:
        available.append("awq")
    if AUTOAWQ_AVAILABLE:
        available.append("autoawq")
    # Add more as they're implemented

    return available


def print_quantization_status() -> None:
    """Print the status of all quantization libraries."""
    print("=" * 60)
    print("Quantization Libraries Status")
    print("=" * 60)
    print(
        f"AWQ (transformers):  {'✓ Available' if AWQ_AVAILABLE else '✗ Not available'}"
    )
    print(
        f"AutoAWQ (direct):    {'✓ Available' if AUTOAWQ_AVAILABLE else '✗ Not available'}"
    )
    # Add more as they're implemented
    print("=" * 60)
