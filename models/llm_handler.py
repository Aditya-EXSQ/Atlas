import json
from threading import Thread
from typing import Dict, Iterator, List

import requests
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


class BaseLLM:
    """Base class for LLM handlers."""

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate streaming response from messages.

        Args:
            messages: List of conversation messages [{"role": "user/assistant", "content": "..."}]
            **kwargs: Additional generation parameters

        Yields:
            Generated text tokens
        """
        raise NotImplementedError


class TransformersLLM(BaseLLM):
    """
    LLM handler using HuggingFace Transformers with 4-bit quantization.
    """

    def __init__(self, config: dict):
        """
        Initialize Transformers model.

        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        model_name = config["model_name"]
        device = config.get("device", "cuda")  # noqa: F841
        load_in_4bit = config.get("load_in_4bit", True)

        print(f"Loading Transformers model: {model_name}")

        # Configure 4-bit quantization
        if load_in_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("Using 4-bit quantization")
        else:
            quantization_config = None
            print("Using full precision")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "torch_dtype": torch.float16
            if torch.cuda.is_available()
            else torch.float32,
        }

        # Add Flash Attention 2 if requested
        if config.get("use_flash_attention_2", False):
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        print(f"Model loaded on: {self.model.device}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate streaming response using TextIteratorStreamer.

        Args:
            messages: List of conversation messages
            **kwargs: Override generation parameters

        Yields:
            Generated text tokens
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generation parameters
        gen_params = {
            "max_new_tokens": kwargs.get(
                "max_new_tokens", self.config.get("max_new_tokens", 512)
            ),
            "temperature": kwargs.get(
                "temperature", self.config.get("temperature", 0.7)
            ),
            "top_p": kwargs.get("top_p", self.config.get("top_p", 0.9)),
            "top_k": kwargs.get("top_k", self.config.get("top_k", 50)),
            "repetition_penalty": kwargs.get(
                "repetition_penalty", self.config.get("repetition_penalty", 1.1)
            ),
            "do_sample": True,
        }

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation in separate thread
        generation_kwargs = {**inputs, **gen_params, "streamer": streamer}
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream tokens
        for text in streamer:
            yield text

        thread.join()


class OllamaLLM(BaseLLM):
    """
    LLM handler using Ollama REST API.
    """

    def __init__(self, config: dict):
        """
        Initialize Ollama client.

        Args:
            config: Configuration dictionary containing Ollama settings
        """
        self.config = config
        self.model_name = config["model_name"]
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.api_url = f"{self.base_url}/api/chat"

        print(f"Initialized Ollama client for model: {self.model_name}")
        print(f"Base URL: {self.base_url}")

        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✓ Successfully connected to Ollama")
            else:
                print(
                    f"⚠ Warning: Ollama connection returned status {response.status_code}"
                )
        except Exception as e:
            print(f"⚠ Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running with: ollama serve")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate streaming response using Ollama API.

        Args:
            messages: List of conversation messages
            **kwargs: Override generation parameters

        Yields:
            Generated text tokens
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": kwargs.get(
                    "temperature", self.config.get("temperature", 0.7)
                ),
                "top_p": kwargs.get("top_p", self.config.get("top_p", 0.9)),
                "num_predict": kwargs.get(
                    "max_tokens", self.config.get("max_tokens", 512)
                ),
            },
        }

        try:
            response = requests.post(
                self.api_url, json=payload, stream=True, timeout=60
            )
            response.raise_for_status()

            # Stream response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            if content:
                                yield content

                        # Check if generation is done
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            yield "[Error: Could not connect to Ollama API. Is Ollama running?]"


def create_llm(config: dict) -> BaseLLM:
    """
    Factory function to create LLM based on backend configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        LLM instance (TransformersLLM or OllamaLLM)
    """
    backend = config.get("backend", "ollama").lower()

    if backend == "transformers":
        return TransformersLLM(config["transformers"])
    elif backend == "ollama":
        return OllamaLLM(config["ollama"])
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Choose 'transformers' or 'ollama'"
        )
