"""
Tests for Atlas LLMClient (client.py).
"""

import pytest

from atlas.client import LLMClient
from atlas.config import ProviderConfig
from atlas.core.base import BaseProvider
from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)
from atlas.core.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Fake provider for isolated testing
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            text=f"echo: {request.prompt}",
            model=self.model,
            provider="fake",
            finish_reason="stop",
        )

    async def stream(self, request: GenerateRequest):
        for word in request.prompt.split():
            yield StreamChunk(text=word + " ", model=self.model, provider="fake")

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        texts = (
            request.input if isinstance(request.input, list) else [request.input]
        )
        return EmbeddingResponse(
            embeddings=[[0.1] * 3 for _ in texts],
            model=self.model,
            provider="fake",
        )


@pytest.fixture(autouse=True)
def _register_fake():
    saved = dict(ProviderRegistry._providers)
    ProviderRegistry.add("fake", FakeProvider)
    yield
    ProviderRegistry._providers = saved


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate():
    client = LLMClient(provider="fake", model="test-model")
    resp = await client.generate("hello world")
    assert resp.text == "echo: hello world"
    assert resp.model == "test-model"
    assert resp.provider == "fake"


@pytest.mark.asyncio
async def test_stream():
    client = LLMClient(provider="fake", model="test-model")
    chunks = []
    async for chunk in client.stream("hello world"):
        chunks.append(chunk.text)
    assert "".join(chunks).strip() == "hello world"


@pytest.mark.asyncio
async def test_embeddings():
    client = LLMClient(provider="fake", model="test-model")
    resp = await client.embeddings("hello")
    assert len(resp.embeddings) == 1
    assert len(resp.embeddings[0]) == 3


@pytest.mark.asyncio
async def test_embeddings_list():
    client = LLMClient(provider="fake", model="test-model")
    resp = await client.embeddings(["a", "b"])
    assert len(resp.embeddings) == 2


def test_list_providers():
    providers = LLMClient.list_providers()
    assert "fake" in providers


def test_repr():
    client = LLMClient(provider="fake", model="m")
    assert "fake" in repr(client)
    assert "m" in repr(client)


def test_missing_provider_raises():
    with pytest.raises(Exception):
        LLMClient(provider="nonexistent", model="m")


def test_missing_args_raises():
    with pytest.raises(ValueError):
        LLMClient()


def test_from_config():
    cfg = ProviderConfig(provider="fake", model="cfg-model")
    client = LLMClient(config=cfg)
    assert client.provider_name == "fake"
    assert client.model_name == "cfg-model"
