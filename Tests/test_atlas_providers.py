"""
Tests for Atlas provider adapters.

These tests verify that the built-in provider classes are properly
registered and that they conform to the BaseProvider interface.
They do NOT require running backend servers – network calls are not made.
"""

import os

import pytest

from atlas.core.base import BaseProvider
from atlas.core.exceptions import NotSupportedError
from atlas.core.registry import ProviderRegistry


def test_builtin_providers_registered():
    """All three built-in providers should be registered after import."""
    import atlas.providers  # noqa: F401

    names = ProviderRegistry.list_providers()
    assert "ollama" in names
    assert "vllm" in names
    assert "huggingface" in names


def test_ollama_is_base_provider():
    from atlas.providers.ollama import OllamaProvider

    assert issubclass(OllamaProvider, BaseProvider)


def test_vllm_is_base_provider():
    from atlas.providers.vllm import VLLMProvider

    assert issubclass(VLLMProvider, BaseProvider)


def test_huggingface_is_base_provider():
    from atlas.providers.huggingface import HuggingFaceProvider

    assert issubclass(HuggingFaceProvider, BaseProvider)


def test_ollama_instantiation():
    from atlas.providers.ollama import OllamaProvider

    p = OllamaProvider(model="llama3", base_url="http://localhost:11434")
    assert p.model == "llama3"


def test_vllm_instantiation():
    from atlas.providers.vllm import VLLMProvider

    p = VLLMProvider(model="phi-2", base_url="http://localhost:8000/v1")
    assert p.model == "phi-2"


def test_huggingface_instantiation():
    from atlas.providers.huggingface import HuggingFaceProvider

    p = HuggingFaceProvider(model="gpt2")
    assert p.model == "gpt2"


def test_huggingface_uses_env_token():
    from atlas.providers.huggingface import HuggingFaceProvider

    os.environ["HF_TOKEN"] = "test-tok-123"
    try:
        p = HuggingFaceProvider(model="gpt2")
        assert p.api_token == "test-tok-123"
    finally:
        del os.environ["HF_TOKEN"]


@pytest.mark.asyncio
async def test_vllm_embeddings_not_supported():
    from atlas.core.models import EmbeddingRequest
    from atlas.providers.vllm import VLLMProvider

    p = VLLMProvider(model="phi-2")
    with pytest.raises(NotSupportedError):
        await p.embeddings(EmbeddingRequest(input="hello"))
