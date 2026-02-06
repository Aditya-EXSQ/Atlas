"""
Tests for Atlas data contracts (core/models.py).
"""

from atlas.core.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    StreamChunk,
)


def test_generate_request_defaults():
    req = GenerateRequest(prompt="hello")
    assert req.prompt == "hello"
    assert req.model is None
    assert req.temperature == 0.7
    assert req.top_p == 0.9
    assert req.max_tokens == 512
    assert req.stop is None
    assert req.extra == {}


def test_generate_request_custom():
    req = GenerateRequest(
        prompt="hi",
        model="llama3",
        temperature=0.5,
        top_p=0.8,
        max_tokens=128,
        stop=["###"],
    )
    assert req.model == "llama3"
    assert req.temperature == 0.5
    assert req.stop == ["###"]


def test_generate_response():
    resp = GenerateResponse(
        text="world",
        model="llama3",
        provider="ollama",
        usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        finish_reason="stop",
    )
    assert resp.text == "world"
    assert resp.provider == "ollama"
    assert resp.usage["total_tokens"] == 15


def test_stream_chunk():
    chunk = StreamChunk(text="tok", model="m", provider="p", finish_reason=None)
    assert chunk.text == "tok"
    assert chunk.finish_reason is None


def test_embedding_request_single():
    req = EmbeddingRequest(input="hello")
    assert req.input == "hello"
    assert req.model is None


def test_embedding_request_list():
    req = EmbeddingRequest(input=["a", "b"])
    assert len(req.input) == 2


def test_embedding_response():
    resp = EmbeddingResponse(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        model="embed-model",
        provider="ollama",
    )
    assert len(resp.embeddings) == 2
    assert resp.provider == "ollama"
