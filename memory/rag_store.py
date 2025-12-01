import os
import pickle
from typing import Dict, List, Tuple

import faiss
import numpy as np


class RAGStore:
    """
    FAISS-based vector store for conversation history.
    Stores embedded conversation turns for similarity-based retrieval.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_path: str = "./faiss_index",
        index_file: str = "conversation_index.faiss",
        metadata_file: str = "conversation_metadata.pkl",
    ):
        """
        Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embeddings
            index_path: Directory to store index files
            index_file: FAISS index filename
            metadata_file: Metadata pickle filename
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index_file = os.path.join(index_path, index_file)
        self.metadata_file = os.path.join(index_path, metadata_file)

        # Create index directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)

        # Initialize or load FAISS index
        self.index = None
        self.metadata: List[Dict[str, str]] = []
        self._initialize_index()

    def _initialize_index(self):
        """
        Initialize or load existing FAISS index.
        """
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            # Load existing index
            print(f"Loading existing FAISS index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)

            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)

            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            # Create new index (IndexFlatIP for inner product / cosine similarity)
            print(f"Creating new FAISS index with dimension {self.embedding_dim}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = []

    def add_turn(self, embedding: np.ndarray, turn: Dict[str, str]):
        """
        Add a conversation turn to the RAG store.

        Args:
            embedding: Embedding vector for the turn (normalized)
            turn: Turn metadata dictionary
        """
        # Ensure embedding is 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Add to FAISS index
        self.index.add(embedding.astype("float32"))

        # Add metadata
        self.metadata.append(turn)

        print(f"Added turn to RAG store. Total vectors: {self.index.ntotal}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for similar conversation turns.

        Args:
            query_embedding: Query embedding vector (normalized)
            top_k: Number of results to return

        Returns:
            List of tuples (turn_metadata, similarity_score)
        """
        if self.index.ntotal == 0:
            return []

        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Limit top_k to available vectors
        k = min(top_k, self.index.ntotal)

        # Search
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))

        return results

    def save(self):
        """
        Save FAISS index and metadata to disk.
        """
        if self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_file)

            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.metadata, f)

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

    def clear(self):
        """
        Clear all vectors and metadata from the store.
        """
        self.index.reset()
        self.metadata.clear()
        print("Cleared RAG store")
