# ChatBot with Memory & RAG

A multi-turn conversational chatbot with intelligent memory management using FAISS-based RAG (Retrieval-Augmented Generation). Supports both HuggingFace Transformers and Ollama backends with streaming responses.

## Features

✨ **Key Features:**
- 🧠 **Conversation Memory**: Maintains last 5 turns in active memory
- 📚 **RAG Integration**: Archives older conversations in FAISS for context retrieval
- 🔄 **Streaming Responses**: Real-time token-by-token generation
- 🎛️ **Dual Backend Support**: Switch between HuggingFace Transformers and Ollama
- 🚀 **GPU Optimized**: 4-bit quantization for Transformers (fits in 12GB VRAM)
- 💾 **Persistent Memory**: Saves conversation history across sessions

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Input                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           Conversation Manager                      │
│  - Maintains 5-turn sliding window                  │
│  - Archives older turns                             │
└────────────┬─────────────────────┬──────────────────┘
             │                     │
             │ (Archive)           │ (Active Window)
             ▼                     ▼
┌─────────────────────┐  ┌────────────────────────────┐
│   FAISS RAG Store   │  │      LLM Handler           │
│  - Vector search    │  │  - Transformers (4-bit)    │
│  - Similarity match │  │  - Ollama API              │
│  - Context retrieval│  │  - Streaming generation    │
└─────────┬───────────┘  └────────┬───────────────────┘
          │                       │
          │ (Context)             │
          └───────────┬───────────┘
                      ▼
            ┌─────────────────────┐
            │  Streaming Response │
            └─────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 12 (for GPU support)
- RTX 3080ti or similar GPU (12GB+ VRAM recommended)

### Step 1: Install Dependencies

```bash
# Install PyTorch with CUDA 12 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Step 2: Backend Setup

#### Option A: Using Ollama (Recommended for beginners)

1. **Install Ollama:**
   ```bash
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Or download from https://ollama.ai
   ```

2. **Pull a model:**
   ```bash
   ollama pull mistral:7b-instruct
   # or
   ollama pull llama3.1:8b
   # or
   ollama pull gemma:7b
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```

#### Option B: Using HuggingFace Transformers

1. **Login to HuggingFace (for gated models like Llama):**
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Update config** to use Transformers backend (see Configuration)

## Configuration

Edit `config/model_config.yaml` to customize settings:

```yaml
# Switch backend: "transformers" or "ollama"
backend: "ollama"

# Ollama settings
ollama:
  model_name: "mistral:7b-instruct"
  temperature: 0.7
  max_tokens: 512

# Transformers settings  
transformers:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  load_in_4bit: true
  temperature: 0.7
  max_new_tokens: 512

# Memory settings
memory:
  max_active_turns: 5  # Keep last 5 turns in memory
  rag_top_k: 3         # Retrieve top 3 similar past conversations
```

## Usage

### Basic Usage

```bash
# Make sure Ollama is running (if using Ollama backend)
ollama serve

# In another terminal, run the chatbot
python main.py
```

### Example Conversation

```
You: Hi! Can you explain what RAG is?
Assistant: RAG stands for Retrieval-Augmented Generation...

[Active turns: 1/5 | Archived: 0 | RAG Store: 0 turns]

You: How does it work?
Assistant: RAG works by combining retrieval and generation...

[Active turns: 2/5 | Archived: 0 | RAG Store: 0 turns]

... (continue chatting) ...

You: [After 6+ turns] Tell me again about RAG
[🔍 Retrieved 3 similar past conversations]
Assistant: [Response enhanced with past context about RAG]

[Active turns: 5/5 | Archived: 8 | RAG Store: 8 turns]
```

### Commands

- `exit`, `quit`, `bye`, `q` - Exit and save conversation
- `Ctrl+C` - Interrupt and save

## Project Structure

```
ChatBot/
├── config/
│   └── model_config.yaml          # Configuration file
├── models/
│   ├── __init__.py
│   ├── embedding_model.py         # Sentence transformers for RAG
│   └── llm_handler.py             # LLM backends (HF & Ollama)
├── memory/
│   ├── __init__.py
│   ├── conversation_manager.py    # Conversation window manager
│   └── rag_store.py               # FAISS vector store
├── utils/
│   ├── __init__.py
│   └── helpers.py                 # Utility functions
├── faiss_index/                   # FAISS index storage (auto-created)
│   ├── conversation_index.faiss
│   └── conversation_metadata.pkl
├── main.py                        # Main application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## How It Works

### Memory Management

1. **Active Window**: The chatbot maintains the last 5 conversation turns (USER-ASSISTANT pairs) in active memory
2. **Archival**: When a 6th turn arrives, the oldest turn is automatically embedded and stored in FAISS
3. **RAG Retrieval**: For each user query, the system searches FAISS for similar past conversations
4. **Context Injection**: Retrieved context is added to the prompt to enhance responses

### Streaming

Both backends support token-by-token streaming:
- **Transformers**: Uses `TextIteratorStreamer` with threading
- **Ollama**: Uses HTTP streaming API

### GPU Optimization

For Transformers backend:
- 4-bit quantization (NF4) reduces VRAM usage by ~75%
- Flash Attention 2 (optional) for faster inference
- Automatic device mapping for multi-GPU setups

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```yaml
# In config/model_config.yaml, enable 4-bit quantization:
transformers:
  load_in_4bit: true
```

**2. Ollama Connection Error**
```bash
# Make sure Ollama is running:
ollama serve

# Check if model is available:
ollama list
```

**3. FAISS GPU Error**
```bash
# If FAISS-GPU fails, use CPU version:
pip uninstall faiss-gpu
pip install faiss-cpu
```

## Advanced Configuration

### Custom Embedding Model

```yaml
embedding:
  model_name: "sentence-transformers/all-mpnet-base-v2"  # Higher quality
  device: "cuda"
```

### Adjust Memory Window

```yaml
memory:
  max_active_turns: 10  # Keep more turns in memory
  rag_top_k: 5          # Retrieve more context
```

### Ollama Custom Model

```bash
# Create custom model with Modelfile
ollama create my-custom-model -f Modelfile
```

```yaml
ollama:
  model_name: "my-custom-model"
```

## Performance Tips

1. **For 12GB VRAM**: Use 4-bit quantization with 7B models
2. **For 24GB+ VRAM**: Can use full precision or larger models
3. **CPU Only**: Works but slow - use Ollama backend for better experience
4. **Faster Inference**: Enable Flash Attention 2 (requires installation)

## License

MIT License - Feel free to use and modify!

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- HuggingFace Transformers for LLM support
- Ollama for easy local model deployment
- FAISS for efficient vector search
- Sentence Transformers for embeddings
