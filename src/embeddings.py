"""
Embeddings Module
=================
Generates dense vector embeddings using SentenceTransformers models.
"""

from typing import List, Union
import numpy as np


class EmbeddingManager:
    """
    Handles text embedding generation using sentence-transformers models.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of text strings into numpy float32 array.

        Args:
            texts: List of text strings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) and dtype float32.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string into numpy float32 array of shape (1, embedding_dim).
        """
        return self.embed_texts([query])
