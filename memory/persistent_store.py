import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from memory.db_models import SyncMemoryRepository
from utils.timing import measure_time


class PersistentRAGStore:
    """
    Persistent RAG store using FAISS for vectors and SQLAlchemy ORM for metadata.
    Implements normalization for cosine similarity and proper persistence.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_path: str = "./faiss_index",
        index_file: str = "conversation_index.faiss",
        db_file: str = "rag_metadata.db",
    ):
        """
        Initialize Persistent RAG store.

        Args:
            embedding_dim: Dimension of embeddings
            index_path: Directory to store index and DB files
            index_file: FAISS index filename
            db_file: SQLite database filename
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index_file = os.path.join(index_path, index_file)
        self.db_file = os.path.join(index_path, db_file)

        # Create index directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)

        # Initialize FAISS index and ORM repository
        self.index = None
        self.repository = None
        self._initialize_storage()

    def _initialize_storage(self):
        """
        Initialize or load existing FAISS index and database.
        """
        # Initialize ORM repository
        self.repository = SyncMemoryRepository(self.db_file)
        self.repository.initialize()

        # Initialize or load FAISS index
        if os.path.exists(self.index_file):
            print(f"Loading existing FAISS index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            print(f"Creating new FAISS index with dimension {self.embedding_dim}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Verify consistency
        vector_count = self.index.ntotal
        db_count = self.repository.count_memories()

        if vector_count != db_count:
            print(
                f"⚠ Warning: Index has {vector_count} vectors but DB has {db_count} entries"
            )

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity with IndexFlatIP.

        Args:
            embedding: Input embedding vector

        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding / (norm + 1e-8)  # Add small epsilon to avoid division by zero

    def add(
        self,
        embedding: np.ndarray,
        text: str,
        memory_type: str = "user",
        created_at: Optional[datetime] = None,
    ) -> int:
        """
        Add a memory to the store.

        Args:
            embedding: Embedding vector (will be normalized)
            text: The actual text content
            memory_type: Type of memory ('user' or 'assistant')
            created_at: Timestamp (defaults to now)

        Returns:
            ID of the added memory
        """
        # Normalize embedding
        normalized_emb = self._normalize(embedding)

        # Ensure embedding is 2D array
        if normalized_emb.ndim == 1:
            normalized_emb = normalized_emb.reshape(1, -1)

        # Add to FAISS
        self.index.add(normalized_emb.astype("float32"))

        # Add to database using ORM
        memory = self.repository.add_memory(text, memory_type, created_at)

        print(
            f"Added {memory_type} memory (ID: {memory.id}). Total: {self.index.ntotal}"
        )

        return memory.id

    def search(
        self, query_embedding: np.ndarray, top_k: int = 50
    ) -> List[Tuple[Dict[str, any], float]]:
        """
        Search for similar memories.

        Args:
            query_embedding: Query embedding vector (will be normalized)
            top_k: Number of results to return

        Returns:
            List of tuples (memory_dict, similarity_score)
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query
        normalized_query = self._normalize(query_embedding)

        # Ensure query is 2D array
        if normalized_query.ndim == 1:
            normalized_query = normalized_query.reshape(1, -1)

        # Limit top_k to available vectors
        k = min(top_k, self.index.ntotal)

        # Search FAISS
        with measure_time("RAG search time"):
            scores, indices = self.index.search(normalized_query.astype("float32"), k)

        # Fetch metadata from database using ORM
        results = []

        for idx, score in zip(indices[0], scores[0]):
            # FAISS IDs are 0-indexed, SQLite IDs are 1-indexed
            db_id = int(idx) + 1

            memory = self.repository.get_memory_by_id(db_id)
            if memory:
                results.append((memory.to_dict(), float(score)))

        return results

    def save(self):
        """
        Save FAISS index to disk. Database is auto-committed by ORM.
        """
        if self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_file)
            print(
                f"Saved FAISS index with {self.index.ntotal} vectors to {self.index_file}"
            )

    def get_store_size(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Number of stored vectors
        """
        return self.index.ntotal

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the store.

        Returns:
            Dictionary with vector_count and db_count
        """
        db_count = self.repository.count_memories()
        return {"vector_count": self.index.ntotal, "db_count": db_count}

    def clear(self):
        """
        Clear all vectors and metadata from the store.
        """
        self.index.reset()
        self.repository.delete_all_memories()
        print("Cleared RAG store")

    def close(self):
        """
        Close the database connection.
        """
        if self.repository:
            self.repository.close()

    def __del__(self):
        """
        Cleanup on deletion.
        """
        self.close()
