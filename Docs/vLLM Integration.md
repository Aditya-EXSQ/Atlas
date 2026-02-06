# vLLM Integration - Documentation

## Overview

The `vLLM` class has been implemented using the **OpenAI-compatible API server approach** for true token-level streaming. This provides high-performance inference with comprehensive configuration options.

## Architecture

```
vLLM Server (Background Process)
        ↓
OpenAI-Compatible REST API
        ↓
OpenAI Python Client
        ↓
vLLM Class (models/llm_handler.py)
        ↓
Your Application
```

## Features Implemented

### ✅ True Token-Level Streaming
- Real-time token-by-token generation via OpenAI streaming API
- No simulated streaming - actual async streaming from vLLM server

### ✅ Automatic Chat Template Fallback
- **Smart Error Handling**: Automatically detects when models lack chat templates (e.g., `microsoft/phi-2`)
- **Seamless Fallback**: Falls back from chat completions to completions endpoint
- **Custom Prompt Formatting**: Formats multi-turn conversations for models without templates
- **Configurable**: Can force completions endpoint via `use_completions` parameter

### ✅ Quantization Support
- **AWQ** (Activation-aware Weight Quantization)
- **GPTQ** (Generative Pre-trained Transformer Quantization)
- **SqueezeLLM**
- Pre-quantized models only (not on-the-fly quantization)

### ✅ Device & Performance Configuration
- Tensor parallelism for multi-GPU setups
- GPU memory utilization control (default: 90%)
- Data type selection (auto, float16, bfloat16, float32)
- CPU swap space for memory offloading
- CUDA graphs (can disable for debugging)

### ✅ Server Management Utilities
- `start_server()` - Start vLLM server as subprocess
- `stop_server()` - Gracefully stop server
- `start_server_standalone()` - Static method for standalone usage
- Automatic server health checking

### ✅ Generation Parameters
- Temperature, top_p sampling
- Presence penalty, frequency penalty
- Max tokens, stop sequences
- Full OpenAI ChatCompletion API compatibility

## Usage

### Option 1: Manual Server Start (RECOMMENDED)

```bash
# Terminal 1: Start vLLM server
conda activate chatbot
vllm serve microsoft/phi-2 \\
    --port 8000 \\
    --trust-remote-code \\
    --gpu-memory-utilization 0.9

    OR
vllm serve microsoft/phi-2 --port 8000 --enforce-eager --gpu-memory-utilization 0.8

# Terminal 2: Run your application
conda activate chatbot
python your_app.py
```

```python
# your_app.py
from models.llm_handler import create_llm

config = {
    "backend": "vllm",
    "vllm": {
        "model_name": "microsoft/phi-2",
        "port": 8000,
        "temperature": 0.5,
        "max_tokens": 200,
    }
}

llm = create_llm(config)

messages = [{"role": "user", "content": "Write a Python hello world"}]
for token in llm.generate(messages):
    print(token, end="", flush=True)
```

### Option 2: Programmatic Server Management

```python
from models.llm_handler import vLLM

config = {
    "model_name": "microsoft/phi-2",
    "port": 8000,
    "trust_remote_code": True,
    "gpu_memory_utilization": 0.9,
}

# Create instance
llm = vLLM(config)

# Start server in background
llm.start_server()

# Use the model
messages = [{"role": "user", "content": "Hello!"}]
for token in llm.generate(messages):
    print(token, end="", flush=True)

# Clean up
llm.stop_server()
```

### Option 3: Standalone Server Management

```python
from models.llm_handler import vLLM

config = {"model_name": "microsoft/phi-2", "port": 8000}

# Start server without creating vLLM instance
process = vLLM.start_server_standalone(config)

# ... your code ...

# Stop when done
process.terminate()
```

## Configuration Reference

### Complete Configuration Example

```yaml
vllm:
  # Required
  model_name: "microsoft/phi-2"
  
  # Server Configuration
  port: 8000
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  served_model_name: null  # Optional override
  
  # Performance & Device
  trust_remote_code: true
  dtype: "auto"  # "auto", "half", "float16", "bfloat16", "float32"
  tensor_parallel_size: 1  # >1 for multi-GPU
  gpu_memory_utilization: 0.9  # 0.0-1.0
  
  # Quantization (for pre-quantized models)
  quantization: null  # "awq", "gptq", "squeezellm", or null
  
  # Advanced Options
  max_model_len: null  # Override context length
  download_dir: null  # Custom model download path
  swap_space: null  # CPU swap in GiB
  enforce_eager: false  # Disable CUDA graphs
  seed: null  # Random seed
  
  # Generation Parameters
  temperature: 0.7
  top_p: 0.9
  max_tokens: 512
  presence_penalty: 0.0
  frequency_penalty: 0.0
```

### Parameter Descriptions

#### Server Configuration
- **model_name**: HuggingFace model ID or local path (required)
- **port**: Server port (default: 8000)
- **base_url**: Full API URL (auto-generated from port)
- **api_key**: API key ("EMPTY" for local server)
- **served_model_name**: Model name exposed in API (defaults to model_name)

#### Performance & Device
- **trust_remote_code**: Allow custom model code execution
- **dtype**: Model precision (auto/float16/bfloat16/float32)
- **tensor_parallel_size**: Number of GPUs for tensor parallelism
- **gpu_memory_utilization**: Fraction of GPU memory (0.0-1.0)

#### Quantization
- **quantization**: Use pre-quantized models (awq/gptq/squeezellm)
  - Model must already be quantized in specified format
  - Not for on-the-fly quantization

#### Advanced Options
- **max_model_len**: Override max context length
- **download_dir**: Where to download model files
- **swap_space**: CPU memory for offloading (in GiB)
- **enforce_eager**: Disable CUDA graphs (for debugging)
- **seed**: Random seed for reproducibility

#### Generation Parameters
- **temperature**: Sampling temperature (0.0-2.0)
- **top_p**: Nucleus sampling threshold
- **max_tokens**: Maximum tokens to generate
- **presence_penalty**: Penalize repeated topics (-2.0 to 2.0)
- **frequency_penalty**: Penalize repeated tokens (-2.0 to 2.0)

## Testing

Run the test script:

```bash
conda activate chatbot

# Option 1: With manual server start (recommended)
# Terminal 1:
vllm serve microsoft/phi-2 --port 8000 --trust-remote-code

# Terminal 2:
python test_vllm.py

# Option 2: Programmatic server management
# (uncomment demo_server_management() in test_vllm.py)
python test_vllm.py
```

## Performance Tips

### For Small Models (< 7B)
```yaml
vllm:
  model_name: "microsoft/phi-2"
  gpu_memory_utilization: 0.9
  dtype: "float16"
```

### For Large Models (7B-13B)
```yaml
vllm:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  quantization: "awq"  # Use pre-quantized model
  gpu_memory_utilization: 0.95
  dtype: "auto"
```

### Multi-GPU Setup
```yaml
vllm:
  model_name: "meta-llama/Meta-Llama-3.1-70B-Instruct"
  tensor_parallel_size: 4  # 4 GPUs
  gpu_memory_utilization: 0.95
```

## Troubleshooting

### Server Connection Failed
```
⚠ Warning: Could not connect to vLLM server
```
**Solution**: Start the vLLM server first
```bash
vllm serve microsoft/phi-2 --port 8000 --trust-remote-code
```

### Model Not Found
```
✗ Error loading vLLM model: Model not found
```
**Solution**: Check model name or provide local path

### Out of Memory
```
CUDA out of memory
```
**Solution**: 
- Reduce `gpu_memory_utilization` to 0.8
- Use quantized model (awq/gptq)
- Use smaller model or multi-GPU setup

### Import Error
```
ModuleNotFoundError: No module named 'openai'
```
**Solution**:
```bash
conda activate chatbot
pip install openai
```

### Chat Template Error (Auto-Fixed)
```
Error code: 400 - {'error': {'message': 'default chat template is no longer allowed...'}}
```
**Note**: This error is automatically handled! The vLLM class detects this and falls back to the completions endpoint.

If you see this error, the fallback mechanism should work automatically. However, if you want to force the completions endpoint from the start, add to your config:
```yaml
vllm:
  use_completions: true  # Skip chat completions, use completions directly
```

**Models affected**: `microsoft/phi-2`, and other models without built-in chat templates.

## Comparison with Direct vLLM Usage

### Previous Implementation (Direct LLM)
- ❌ No true streaming (word-level chunks)
- ❌ Synchronous generation only
- ✅ Simple setup
- ✅ No server management

### New Implementation (OpenAI API)
- ✅ True token-level streaming
- ✅ Async-capable via OpenAI client
- ✅ Production-ready architecture
- ✅ Compatible with OpenAI ecosystem
- ⚠️ Requires server management

## Future Enhancements

Potential features for later:
- LoRA adapter support
- Prefix caching configuration
- Speculative decoding
- Guided generation (JSON mode, regex)
- Multi-modal support
- Batch processing utilities

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [vLLM Serving Guide](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
