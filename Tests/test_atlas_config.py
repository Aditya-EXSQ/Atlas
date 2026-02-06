"""
Tests for Atlas configuration (config.py).
"""

import os

import pytest

from atlas.config import ProviderConfig


def test_from_dict():
    data = {
        "provider": "ollama",
        "model": "llama3",
        "base_url": "http://localhost:11434",
        "timeout": 60.0,
        "custom_opt": True,
    }
    cfg = ProviderConfig.from_dict(data)
    assert cfg.provider == "ollama"
    assert cfg.model == "llama3"
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.timeout == 60.0
    assert cfg.extra["custom_opt"] is True


def test_from_dict_defaults():
    cfg = ProviderConfig.from_dict({"provider": "vllm", "model": "phi-2"})
    assert cfg.timeout == 120.0
    assert cfg.base_url is None
    assert cfg.api_key is None


def test_from_env():
    os.environ["TEST_ATLAS_PROVIDER"] = "ollama"
    os.environ["TEST_ATLAS_MODEL"] = "llama3"
    os.environ["TEST_ATLAS_BASE_URL"] = "http://host:1234"
    os.environ["TEST_ATLAS_TIMEOUT"] = "30"
    try:
        cfg = ProviderConfig.from_env(prefix="TEST_ATLAS")
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3"
        assert cfg.base_url == "http://host:1234"
        assert cfg.timeout == 30.0
    finally:
        for k in [
            "TEST_ATLAS_PROVIDER",
            "TEST_ATLAS_MODEL",
            "TEST_ATLAS_BASE_URL",
            "TEST_ATLAS_TIMEOUT",
        ]:
            os.environ.pop(k, None)


def test_from_env_missing():
    with pytest.raises(ValueError, match="must be set"):
        ProviderConfig.from_env(prefix="MISSING_PREFIX")
