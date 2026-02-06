"""Auto-import all built-in providers so they register themselves."""

from atlas.providers.huggingface import HuggingFaceProvider  # noqa: F401
from atlas.providers.ollama import OllamaProvider  # noqa: F401
from atlas.providers.vllm import VLLMProvider  # noqa: F401

__all__ = ["OllamaProvider", "VLLMProvider", "HuggingFaceProvider"]
