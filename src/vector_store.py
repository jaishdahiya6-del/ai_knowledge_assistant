"""
Vector Store Module
===================
Simple FAISS-based vector store for indexing text chunks, saving/loading index,
and performing similarity search with metadata retrieval.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

from src.embeddings import EmbeddingManager


class VectorStore:
    """
    FAISS vector store for semantic similarity search over document chunks.
    """

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[Dict[str, Any]] = []  # Stores [{"text": ..., "metadata": ...}]

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Generates embeddings for chunk texts and adds them to the FAISS index.

        Args:
            chunks: List of chunk dicts containing 'text' and 'metadata'.
        """
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)

        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches for top_k most similar document chunks to the query string.

        Returns:
            List of tuples: (chunk_dict, distance_score)
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.embedding_manager.embed_query(query)
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        return results

    def save(self, directory: str = "data/vector_store") -> None:
        """
        Saves the FAISS index and chunk metadata store to specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        index_file = os.path.join(directory, "index.faiss")
        metadata_file = os.path.join(directory, "chunks.json")

        if self.index is not None:
            faiss.write_index(self.index, index_file)

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def load(self, directory: str = "data/vector_store") -> bool:
        """
        Loads FAISS index and chunk metadata store from directory.

        Returns:
            bool: True if successfully loaded, False otherwise.
        """
        index_file = os.path.join(directory, "index.faiss")
        metadata_file = os.path.join(directory, "chunks.json")

        if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
            return False

        self.index = faiss.read_index(index_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        return True

    def clear(self) -> None:
        """Clears index and stored chunks."""
        self.index = None
        self.chunks = []
