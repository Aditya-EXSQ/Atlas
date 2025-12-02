from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils.timing import measure_time


class EmbeddingModel:
    """
    Handles text embeddings for RAG using sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name} on {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]

        with measure_time("Embedding generation time"):
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
            )

        return embeddings

    def get_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim
