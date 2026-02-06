from typing import Dict, List, Tuple

import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved passages using a Cross-Encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda",
    ):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the Cross-Encoder model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading Reranker model {model_name} on {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self, query: str, candidates: List[Dict[str, str]], top_k: int = 5
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Rerank a list of candidate passages based on the query.

        Args:
            query: The search query
            candidates: List of candidate dictionaries (must contain 'content' or 'text' field)
            top_k: Number of top results to return

        Returns:
            List of tuples (candidate, score) sorted by score descending
        """
        if not candidates:
            return []

        # Prepare pairs for cross-encoder
        # Assuming candidates have 'content' field based on previous code, but let's be flexible
        passages = [c.get("content", c.get("text", "")) for c in candidates]
        model_inputs = [[query, passage] for passage in passages]

        # Predict scores
        scores = self.model.predict(model_inputs)

        # Combine with candidates
        results = list(zip(candidates, scores))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]
