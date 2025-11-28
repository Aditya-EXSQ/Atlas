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

import sys
import os
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import EmbeddingModel, create_llm
from memory import ConversationManager, RAGStore
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
        print_colored("\n[1/4] Loading Embedding Model...", "system")
        embedding_config = self.config['embedding']
        self.embedding_model = EmbeddingModel(
            model_name=embedding_config['model_name'],
            device=embedding_config.get('device', 'cuda')
        )
        
        # Initialize RAG store
        print_colored("\n[2/4] Initializing RAG Store...", "system")
        faiss_config = self.config['faiss']
        self.rag_store = RAGStore(
            embedding_dim=self.embedding_model.get_dimension(),
            index_path=faiss_config['index_path'],
            index_file=faiss_config['index_file'],
            metadata_file=faiss_config['metadata_file']
        )
        
        # Initialize conversation manager
        print_colored("\n[3/4] Initializing Conversation Manager...", "system")
        memory_config = self.config['memory']
        self.conversation_manager = ConversationManager(
            max_active_turns=memory_config['max_active_turns']
        )
        self.rag_top_k = memory_config.get('rag_top_k', 3)
        
        # Initialize LLM
        print_colored("\n[4/4] Loading LLM...", "system")
        backend = self.config.get('backend', 'ollama')
        print_colored(f"Using backend: {backend.upper()}", "system")
        self.llm = create_llm(self.config)
        
        print_colored("\n" + "=" * 60, "system")
        print_colored("✓ ChatBot initialized successfully!", "system")
        print_colored("=" * 60, "system")
    
    def _retrieve_rag_context(self, user_message: str) -> str:
        """
        Retrieve relevant context from RAG store.
        
        Args:
            user_message: Current user message
            
        Returns:
            Formatted context string from RAG
        """
        if self.rag_store.get_store_size() == 0:
            return ""
        
        # Embed user message
        query_embedding = self.embedding_model.encode(user_message)
        
        # Search for similar past conversations
        results = self.rag_store.search(query_embedding, top_k=self.rag_top_k)
        
        if not results:
            return ""
        
        # Format context
        context_parts = ["[Relevant past conversation context:]"]
        for turn, score in results:
            context_parts.append(f"- {turn['role']}: {turn['content']} (similarity: {score:.2f})")
        
        return "\n".join(context_parts) + "\n"
    
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
        self.rag_store.add_turn(embedding, turn)
    
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
            print_colored(f"\n[📦 Archived old turn to RAG]", "system")
            self._archive_turn(archived_turn)
        
        # Retrieve RAG context
        rag_context = self._retrieve_rag_context(user_message)
        
        # Prepare messages for LLM
        messages = self.conversation_manager.get_messages_for_llm()
        
        # Add RAG context to the latest user message if available
        if rag_context:
            print_colored(f"\n[🔍 Retrieved {self.rag_top_k} similar past conversations]", "system")
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
            print_colored(f"[📦 Archived old turn to RAG]", "system")
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
                if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
