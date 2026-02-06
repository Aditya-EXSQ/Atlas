"""
Tests for Atlas provider registry (core/registry.py).
"""

import pytest

from atlas.core.base import BaseProvider
from atlas.core.exceptions import ProviderNotFoundError
from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)
from atlas.core.registry import ProviderRegistry


class _DummyProvider(BaseProvider):
    """Minimal concrete provider for testing."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            text="dummy", model=self.model, provider="dummy"
        )

    async def stream(self, request: GenerateRequest):
        yield StreamChunk(text="d", model=self.model, provider="dummy")

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(
            embeddings=[[0.0]], model=self.model, provider="dummy"
        )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(ProviderRegistry._providers)
    yield
    ProviderRegistry._providers = saved


def test_register_decorator():
    @ProviderRegistry.register("test_dec")
    class TestProv(_DummyProvider):
        pass

    assert "test_dec" in ProviderRegistry.list_providers()
    assert ProviderRegistry.get("test_dec") is TestProv


def test_add_runtime():
    ProviderRegistry.add("test_rt", _DummyProvider)
    assert ProviderRegistry.get("test_rt") is _DummyProvider


def test_get_missing():
    with pytest.raises(ProviderNotFoundError) as exc_info:
        ProviderRegistry.get("nonexistent")
    assert "nonexistent" in str(exc_info.value)


def test_list_providers():
    ProviderRegistry.add("a", _DummyProvider)
    ProviderRegistry.add("b", _DummyProvider)
    names = ProviderRegistry.list_providers()
    assert "a" in names
    assert "b" in names


def test_clear():
    ProviderRegistry.add("temp", _DummyProvider)
    ProviderRegistry.clear()
    assert ProviderRegistry.list_providers() == []
