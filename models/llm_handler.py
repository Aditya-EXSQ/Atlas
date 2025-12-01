import json
import subprocess
import time
from threading import Thread
from typing import Dict, Iterator, List, Optional

import requests
import torch
from openai import OpenAI
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
            "dtype": torch.float16
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


class TransformersNoGradLLM(TransformersLLM):
    """
    LLM handler using HuggingFace Transformers with torch.no_grad() during generation.

    This class inherits all functionality from TransformersLLM but wraps the generation
    process in torch.no_grad() context for inference optimization. Useful for benchmarking
    and comparing inference performance.
    """

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate streaming response with torch.no_grad() context.

        Args:
            messages: List of conversation messages
            **kwargs: Override generation parameters

        Yields:
            Generated text tokens
        """
        with torch.no_grad():
            yield from super().generate(messages, **kwargs)


class TransformersInferenceModeLLM(TransformersLLM):
    """
    LLM handler using HuggingFace Transformers with torch.inference_mode() during generation.

    This class inherits all functionality from TransformersLLM but wraps the generation
    process in torch.inference_mode() context for maximum performance optimization.
    inference_mode() is more restrictive than no_grad() and may provide additional
    performance benefits. Useful for benchmarking and comparing inference performance.
    """

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate streaming response with torch.inference_mode() context.

        Args:
            messages: List of conversation messages
            **kwargs: Override generation parameters

        Yields:
            Generated text tokens
        """
        with torch.inference_mode():
            yield from super().generate(messages, **kwargs)


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


class vLLM(BaseLLM):
    """
    LLM handler using vLLM's OpenAI-compatible API server for high-performance inference.

    This implementation uses the vLLM server (started via `vllm serve`) and communicates
    with it using the OpenAI Python client for true token-level streaming.

    Supports:
    - True token-level streaming via OpenAI API
    - Quantization (AWQ, GPTQ, SqueezeLLM)
    - Tensor parallelism for multi-GPU
    - GPU memory utilization control
    - KV cache optimization
    - Server management utilities
    """

    def __init__(self, config: dict):
        """
        Initialize vLLM OpenAI API client.

        Args:
            config: Configuration dictionary containing vLLM settings
                Required:
                    - model_name: Model identifier (HuggingFace model ID or path)
                Optional:
                    - base_url: API server URL (default: "http://localhost:8000/v1")
                    - port: Server port (default: 8000)
                    - api_key: API key (default: "EMPTY" for local server)
                    - dtype: Data type ("auto", "half", "float16", "bfloat16", "float32")
                    - quantization: Quantization method ("awq", "gptq", "squeezellm", None)
                    - tensor_parallel_size: Number of GPUs for tensor parallelism
                    - gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
                    - trust_remote_code: Whether to trust remote code
                    - max_model_len: Maximum sequence length
                    - download_dir: Directory to download model files
                    - enforce_eager: Disable CUDA graphs for debugging
                    - swap_space: CPU swap space size in GiB
                    - seed: Random seed for reproducibility
                    - served_model_name: Model name to use in API (defaults to model_name)
        """
        self.config = config
        self.model_name = config["model_name"]
        self.port = config.get("port", 8000)
        self.base_url = config.get("base_url", f"http://localhost:{self.port}/v1")
        self.api_key = config.get("api_key", "EMPTY")
        self.served_model_name = config.get("served_model_name", self.model_name)
        self.server_process: Optional[subprocess.Popen] = None

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        print(f"Initialized vLLM OpenAI client for model: {self.model_name}")
        print(f"API Base URL: {self.base_url}")
        print(f"Served model name: {self.served_model_name}")

        # Test connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """
        Test connection to vLLM server.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Try to list models
            models = self.client.models.list()
            print("✓ Successfully connected to vLLM server")
            available_models = [model.id for model in models.data]
            print(f"Available models: {available_models}")
            return True
        except Exception as e:
            print(f"⚠ Warning: Could not connect to vLLM server: {e}")
            print("Make sure the vLLM server is running.")
            print(
                'Start it with: python -c "from models.llm_handler import vLLM; vLLM.start_server({...config...})"'
            )
            return False

    def start_server(self) -> subprocess.Popen:
        """
        Start the vLLM server as a background process.

        Returns:
            The subprocess.Popen object for the server process

        Note:
            This is a utility method that must be called manually before using the class.
            The server runs as a separate process and can be stopped with stop_server().
        """
        if self.server_process is not None:
            print("⚠ Server process already running")
            return self.server_process

        print(f"Starting vLLM server for model: {self.model_name}")

        # Build command line arguments
        cmd = [
            "vllm",
            "serve",
            self.model_name,
            "--port",
            str(self.port),
        ]

        # Add optional parameters
        if self.config.get("trust_remote_code", True):
            cmd.append("--trust-remote-code")

        if "dtype" in self.config and self.config["dtype"] != "auto":
            cmd.extend(["--dtype", self.config["dtype"]])

        if "quantization" in self.config and self.config["quantization"]:
            cmd.extend(["--quantization", self.config["quantization"]])

        if (
            "tensor_parallel_size" in self.config
            and self.config["tensor_parallel_size"] > 1
        ):
            cmd.extend(
                ["--tensor-parallel-size", str(self.config["tensor_parallel_size"])]
            )

        if "gpu_memory_utilization" in self.config:
            cmd.extend(
                ["--gpu-memory-utilization", str(self.config["gpu_memory_utilization"])]
            )

        if "max_model_len" in self.config:
            cmd.extend(["--max-model-len", str(self.config["max_model_len"])])

        if "download_dir" in self.config:
            cmd.extend(["--download-dir", self.config["download_dir"]])

        if "swap_space" in self.config:
            cmd.extend(["--swap-space", str(self.config["swap_space"])])

        if self.config.get("enforce_eager", False):
            cmd.append("--enforce-eager")

        if "seed" in self.config:
            cmd.extend(["--seed", str(self.config["seed"])])

        if "served_model_name" in self.config:
            cmd.extend(["--served-model-name", self.config["served_model_name"]])

        print(f"Command: {' '.join(cmd)}")

        # Start the server process
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"✓ vLLM server started (PID: {self.server_process.pid})")
            print("⏳ Waiting for server to be ready (this may take a minute)...")

            # Wait for server to be ready
            max_retries = 60
            for i in range(max_retries):
                time.sleep(2)
                try:
                    response = requests.get(
                        f"http://localhost:{self.port}/health", timeout=2
                    )
                    if response.status_code == 200:
                        print("✓ Server is ready!")
                        return self.server_process
                except:
                    if i % 5 == 0:
                        print(f"Still waiting... ({i * 2}s)")
                    continue

            print("⚠ Server did not become ready in expected time")
            return self.server_process

        except Exception as e:
            print(f"✗ Error starting vLLM server: {e}")
            raise

    def stop_server(self):
        """
        Stop the vLLM server process if it was started by this instance.
        """
        if self.server_process is None:
            print("⚠ No server process to stop")
            return

        print(f"Stopping vLLM server (PID: {self.server_process.pid})")
        self.server_process.terminate()

        try:
            self.server_process.wait(timeout=10)
            print("✓ Server stopped successfully")
        except subprocess.TimeoutExpired:
            print("⚠ Server didn't stop gracefully, killing...")
            self.server_process.kill()
            self.server_process.wait()
            print("✓ Server killed")

        self.server_process = None

    @staticmethod
    def start_server_standalone(config: dict) -> subprocess.Popen:
        """
        Static method to start a vLLM server without creating a vLLM instance.

        Args:
            config: Configuration dictionary (same as __init__)

        Returns:
            The subprocess.Popen object for the server process

        Example:
            >>> config = {"model_name": "microsoft/phi-2", "port": 8000}
            >>> process = vLLM.start_server_standalone(config)
            >>> # ... use the server ...
            >>> process.terminate()
        """
        temp_instance = vLLM.__new__(vLLM)
        temp_instance.config = config
        temp_instance.model_name = config["model_name"]
        temp_instance.port = config.get("port", 8000)
        temp_instance.server_process = None
        return temp_instance.start_server()

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        """
        Generate streaming response using vLLM OpenAI API with true token-level streaming.

        Args:
            messages: List of conversation messages [{"role": "user/assistant/system", "content": "..."}]
            **kwargs: Override generation parameters
                - temperature: Sampling temperature (default: 0.7)
                - top_p: Nucleus sampling probability (default: 0.9)
                - max_tokens: Maximum tokens to generate (default: 512)
                - presence_penalty: Presence penalty (default: 0.0)
                - frequency_penalty: Frequency penalty (default: 0.0)
                - stop: List of stop sequences (optional)
                - use_completions: Force use of completions endpoint (default: False)

        Yields:
            Generated text tokens (true token-level streaming)
        """
        # Prepare generation parameters
        gen_params = {
            "stream": True,
            "temperature": kwargs.get(
                "temperature", self.config.get("temperature", 0.7)
            ),
            "top_p": kwargs.get("top_p", self.config.get("top_p", 0.9)),
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 512)),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.config.get("presence_penalty", 0.0)
            ),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.config.get("frequency_penalty", 0.0)
            ),
        }

        # Add optional stop sequences
        if "stop" in kwargs:
            gen_params["stop"] = kwargs["stop"]
        elif "stop" in self.config:
            gen_params["stop"] = self.config["stop"]

        # Try chat completions first, fall back to completions if chat template not available
        use_completions = kwargs.get(
            "use_completions", self.config.get("use_completions", False)
        )

        if not use_completions:
            try:
                # Try using chat completions API
                gen_params["model"] = self.served_model_name
                gen_params["messages"] = messages

                stream = self.client.chat.completions.create(**gen_params)

                # Stream tokens as they arrive
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            yield delta.content
                return

            except Exception as e:
                error_str = str(e)
                # Check if it's a chat template error
                if "chat template" in error_str.lower() or "400" in error_str:
                    # Fall back to completions endpoint
                    use_completions = True
                else:
                    # Other error, raise it
                    print(f"Error during vLLM generation: {e}")
                    yield f"[Error: {str(e)}]"
                    return

        # Use completions endpoint (for models without chat templates)
        if use_completions:
            try:
                # Format messages into a single prompt
                prompt = self._format_messages_to_prompt(messages)

                gen_params["model"] = self.served_model_name
                gen_params["prompt"] = prompt

                # Remove chat-specific parameters
                gen_params.pop("messages", None)

                stream = self.client.completions.create(**gen_params)

                # Stream tokens as they arrive
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        text = chunk.choices[0].text
                        if text:
                            yield text

            except Exception as e:
                print(f"Error during vLLM generation: {e}")
                yield f"[Error: {str(e)}]"

    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a single prompt string for models without chat templates.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted prompt string
        """
        # Check if there's a custom chat template in config
        if "chat_template" in self.config:
            # Use custom template format
            template = self.config["chat_template"]
            return template.format(messages=messages)

        # Default formatting for phi-2 and similar models
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Instruct: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Output: {content}")

        # Add final prompt for assistant to continue
        prompt_parts.append("Output:")

        return "\n".join(prompt_parts)


def create_llm(config: dict) -> BaseLLM:
    """
    Factory function to create LLM based on backend configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        LLM instance (TransformersLLM, TransformersNoGradLLM, TransformersInferenceModeLLM,
                      OllamaLLM, or vLLM)
    """
    backend = config.get("backend", "ollama").lower()

    if backend == "transformers":
        return TransformersLLM(config["transformers"])
    elif backend == "transformers_no_grad":
        return TransformersNoGradLLM(config["transformers_no_grad"])
    elif backend == "transformers_inference_mode":
        return TransformersInferenceModeLLM(config["transformers_inference_mode"])
    elif backend == "ollama":
        return OllamaLLM(config["ollama"])
    elif backend == "vllm":
        return vLLM(config["vllm"])
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Choose 'transformers', 'transformers_no_grad', "
            f"'transformers_inference_mode', 'ollama', or 'vllm'"
        )
