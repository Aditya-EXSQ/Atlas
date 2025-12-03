#!/usr/bin/env python3
"""
Multi-Turn Conversational ChatBot with Memory and RAG
======================================================
A chatbot with conversation memory management using FAISS-based RAG.
Supports both HuggingFace Transformers and Ollama backends.

Features:
- Maintains last 5 conversation turns in active memory
- Archives older conversations in FAISS vector store
- Retrieves relevant past context using RAG
- Streaming response generation
- Supports multiple LLM backends (Transformers/Ollama)
"""

import os
import sys
import traceback
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory import ConversationManager, PersistentRAGStore, Reranker
from models import EmbeddingModel, create_llm
from utils import load_config, print_colored


class ChatBot:
    """
    Main ChatBot class integrating all components.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the chatbot with all components.

        Args:
            config_path: Path to configuration file
        """
        print_colored("=" * 60, "system")
        print_colored("Initializing ChatBot with Memory & RAG...", "system")
        print_colored("=" * 60, "system")

        # Load configuration
        self.config = load_config(config_path)

        # Initialize embedding model
        print_colored("\n[1/5] Loading Embedding Model...", "system")
        embedding_config = self.config["embedding"]
        self.embedding_model = EmbeddingModel(
            model_name=embedding_config["model_name"],
            device=embedding_config.get("device", "cuda"),
        )

        # Initialize RAG store
        print_colored("\n[2/5] Initializing Persistent RAG Store...", "system")
        faiss_config = self.config["faiss"]
        self.rag_store = PersistentRAGStore(
            embedding_dim=self.embedding_model.get_dimension(),
            index_path=faiss_config["index_path"],
            index_file=faiss_config.get("index_file", "conversation_index.faiss"),
            db_file=faiss_config.get("db_file", "rag_metadata.db"),
        )

        # Initialize reranker
        print_colored("\n[3/5] Loading Reranker...", "system")
        reranker_config = self.config.get("reranker", {})
        self.reranker = Reranker(
            model_name=reranker_config.get(
                "model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            device=reranker_config.get("device", "cuda"),
        )

        # Initialize conversation manager
        print_colored("\n[4/5] Initializing Conversation Manager...", "system")
        memory_config = self.config["memory"]
        self.conversation_manager = ConversationManager(
            max_active_turns=memory_config["max_active_turns"]
        )
        self.rag_retrieve_k = memory_config.get("rag_retrieve_k", 50)
        self.rag_rerank_k = memory_config.get("rag_rerank_k", 5)
        self.persist_min_length = memory_config.get("persist_min_length", 15)

        # Initialize LLM
        print_colored("\n[5/5] Loading LLM...", "system")
        backend = self.config.get("backend", "ollama")
        print_colored(f"Using backend: {backend.upper()}", "system")
        self.llm = create_llm(self.config)

        print_colored("\n" + "=" * 60, "system")
        print_colored("✓ ChatBot initialized successfully!", "system")
        print_colored("=" * 60, "system")

    def _retrieve_rag_context(self, user_message: str) -> str:
        """
        Retrieve relevant context from RAG store using retrieve-then-rerank.

        Args:
            user_message: Current user message

        Returns:
            Formatted context string from RAG
        """
        if self.rag_store.get_store_size() == 0:
            return ""

        # Embed user message
        query_embedding = self.embedding_model.encode(user_message)

        # Step 1: Retrieve top_k candidates from FAISS
        candidates = self.rag_store.search(query_embedding, top_k=self.rag_retrieve_k)

        if not candidates:
            return ""

        # Step 2: Rerank using cross-encoder
        reranked_results = self.reranker.rerank(
            query=user_message,
            candidates=[c[0] for c in candidates],  # Extract just the memory dicts
            top_k=self.rag_rerank_k,
        )

        if not reranked_results:
            return ""

        # Format context
        context_parts = ["[Relevant past conversation context:]"]
        for memory, score in reranked_results:
            # Handle both 'content' and 'text' fields
            text = memory.get("content", memory.get("text", ""))
            memory_type = memory.get("type", "unknown")
            context_parts.append(f"- {memory_type}: {text} (relevance: {score:.3f})")

        return "\n".join(context_parts) + "\n"

    def _decide_persist(self, text: str) -> bool:
        """
        Decide whether to persist a message to long-term memory.
        Simple heuristic: message length > threshold and not a greeting.

        Args:
            text: Message text

        Returns:
            True if should persist, False otherwise
        """
        # Skip short messages
        if len(text) < self.persist_min_length:
            return False

        # Skip common greetings
        greetings = {"hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you"}
        if text.lower().strip() in greetings:
            return False

        return True

    def _archive_turn(self, turn: Dict[str, str]):
        """
        Archive a turn to the RAG store.

        Args:
            turn: Turn dictionary to archive
        """
        # Format turn for embedding
        turn_text = self.conversation_manager.format_turn_for_embedding(turn)

        # Embed and store
        embedding = self.embedding_model.encode(turn_text)
        memory_type = turn.get("role", "user")
        self.rag_store.add(embedding, turn["content"], memory_type=memory_type)

    def generate_response(self, user_message: str) -> str:
        """
        Generate a response to user message with streaming.

        Args:
            user_message: User's message

        Returns:
            Complete assistant response
        """
        # Add user message to conversation
        archived_turn = self.conversation_manager.add_turn("user", user_message)

        # Archive if needed
        if archived_turn:
            print_colored("\n[📦 Archived old turn to RAG]", "system")
            self._archive_turn(archived_turn)

        # Persist to long-term memory if appropriate
        if self._decide_persist(user_message):
            # Embed and add current message to long-term store
            embedding = self.embedding_model.encode(user_message)
            self.rag_store.add(embedding, user_message, memory_type="user")

        # Retrieve RAG context
        rag_context = self._retrieve_rag_context(user_message)

        # Prepare messages for LLM
        messages = self.conversation_manager.get_messages_for_llm()

        # Add RAG context to the latest user message if available
        if rag_context:
            print_colored(
                f"\n[🔍 Retrieved top {self.rag_rerank_k} (from {self.rag_retrieve_k}) relevant memories]",
                "system",
            )
            messages[-1]["content"] = rag_context + "\n" + messages[-1]["content"]

        # Generate streaming response
        print_colored("\nAssistant: ", "assistant", "", end="")

        response_parts = []
        try:
            for token in self.llm.generate(messages):
                print_colored(token, "assistant", "", end="")
                sys.stdout.flush()
                response_parts.append(token)
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print_colored(f"\n{error_msg}", "error")
            return error_msg

        print()  # New line after streaming

        # Combine response
        full_response = "".join(response_parts)

        # Add assistant response to conversation
        archived_turn = self.conversation_manager.add_turn("assistant", full_response)

        # Archive if needed
        if archived_turn:
            print_colored("[📦 Archived old turn to RAG]", "system")
            self._archive_turn(archived_turn)

        return full_response

    def chat_loop(self):
        """
        Main interactive chat loop.
        """
        print_colored("\n" + "=" * 60, "system")
        print_colored("ChatBot Ready! Type 'exit', 'quit', or 'bye' to end.", "system")
        print_colored("=" * 60 + "\n", "system")

        try:
            while True:
                # Get user input
                print_colored("You: ", "user", "", end="")
                user_input = input().strip()

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "bye", "q"]:
                    print_colored("\nGoodbye! Saving RAG store...", "system")
                    self.rag_store.save()
                    print_colored("✓ Conversation saved successfully!", "system")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Generate response
                self.generate_response(user_input)

                # Show conversation status
                status = self.conversation_manager.get_conversation_summary()
                rag_size = self.rag_store.get_store_size()
                print_colored(f"\n[{status} | RAG Store: {rag_size} turns]\n", "system")

        except KeyboardInterrupt:
            print_colored("\n\nInterrupted! Saving RAG store...", "system")
            self.rag_store.save()
            print_colored("✓ Conversation saved successfully!", "system")

        except Exception as e:
            print_colored(f"\nError in chat loop: {e}", "error")
            self.rag_store.save()


def main():
    """
    Main entry point for the chatbot application.
    """
    # Parse command line arguments
    config_path = "config/model_config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # Initialize and run chatbot
    try:
        bot = ChatBot(config_path=config_path)
        bot.chat_loop()
    except Exception as e:
        print_colored(f"Fatal error: {e}", "error")

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
