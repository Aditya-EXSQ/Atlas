import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from utils.timing import measure_time


class PersistentRAGStore:
    """
    Persistent RAG store using FAISS for vectors and SQLite for metadata.
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

        # Initialize FAISS index and SQLite DB
        self.index = None
        self.db_conn = None
        self._initialize_storage()

    def _initialize_storage(self):
        """
        Initialize or load existing FAISS index and SQLite database.
        """
        # Initialize SQLite database
        self.db_conn = sqlite3.connect(self.db_file)
        self.db_conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = self.db_conn.cursor()

        # Create table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                type TEXT NOT NULL
            )
        """
        )
        self.db_conn.commit()

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
        cursor.execute("SELECT COUNT(*) FROM memories")
        db_count = cursor.fetchone()[0]

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

        # Add to SQLite
        if created_at is None:
            created_at = datetime.now()

        cursor = self.db_conn.cursor()
        cursor.execute(
            """
            INSERT INTO memories (text, created_at, type)
            VALUES (?, ?, ?)
        """,
            (text, created_at, memory_type),
        )
        self.db_conn.commit()

        memory_id = cursor.lastrowid
        print(
            f"Added {memory_type} memory (ID: {memory_id}). Total: {self.index.ntotal}"
        )

        return memory_id

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

        # Fetch metadata from SQLite
        cursor = self.db_conn.cursor()
        results = []

        for idx, score in zip(indices[0], scores[0]):
            # FAISS IDs are 0-indexed, SQLite IDs are 1-indexed
            db_id = int(idx) + 1

            cursor.execute(
                """
                SELECT id, text, created_at, type
                FROM memories
                WHERE id = ?
            """,
                (db_id,),
            )

            row = cursor.fetchone()
            if row:
                memory = {
                    "id": row["id"],
                    "content": row["text"],  # Use 'content' for compatibility
                    "text": row["text"],  # Also provide 'text'
                    "created_at": row["created_at"],
                    "type": row["type"],
                }
                results.append((memory, float(score)))

        return results

    def save(self):
        """
        Save FAISS index to disk. SQLite is auto-committed.
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
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        db_count = cursor.fetchone()[0]

        return {"vector_count": self.index.ntotal, "db_count": db_count}

    def clear(self):
        """
        Clear all vectors and metadata from the store.
        """
        self.index.reset()

        cursor = self.db_conn.cursor()
        cursor.execute("DELETE FROM memories")
        self.db_conn.commit()

        print("Cleared RAG store")

    def close(self):
        """
        Close the database connection.
        """
        if self.db_conn:
            self.db_conn.close()

    def __del__(self):
        """
        Cleanup on deletion.
        """
        self.close()
