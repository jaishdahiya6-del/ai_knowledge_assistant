import os
import tempfile
import unittest
import numpy as np

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.embedding_mgr = EmbeddingManager()
        self.vector_store = VectorStore(embedding_manager=self.embedding_mgr)

        self.sample_chunks = [
            {
                "text": "Gradient descent is an optimization algorithm used to minimize loss in machine learning.",
                "metadata": {"source": "ml_guide.md", "page": 1, "chunk_id": 0}
            },
            {
                "text": "Random Forest is an ensemble learning method using multiple decision trees.",
                "metadata": {"source": "ml_guide.md", "page": 2, "chunk_id": 1}
            },
            {
                "text": "SQLite is a lightweight C-language library that implements a SQL database engine.",
                "metadata": {"source": "db_notes.txt", "page": 1, "chunk_id": 0}
            }
        ]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_embeddings_generator(self):
        embeddings = self.embedding_mgr.embed_texts(["Hello world", "Test sentence"])
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 384)

    def test_add_and_search_documents(self):
        self.vector_store.add_documents(self.sample_chunks)
        self.assertEqual(self.vector_store.index.ntotal, 3)

        results = self.vector_store.search("How does gradient descent work?", top_k=2)
        self.assertEqual(len(results), 2)

        top_chunk, dist = results[0]
        self.assertIn("Gradient descent", top_chunk["text"])

    def test_save_and_load_vector_store(self):
        self.vector_store.add_documents(self.sample_chunks)
        save_dir = os.path.join(self.tmp_dir.name, "vector_db")

        self.vector_store.save(save_dir)

        new_store = VectorStore(embedding_manager=self.embedding_mgr)
        loaded = new_store.load(save_dir)

        self.assertTrue(loaded)
        self.assertEqual(new_store.index.ntotal, 3)
        self.assertEqual(len(new_store.chunks), 3)

        results = new_store.search("SQLite database", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("SQLite", results[0][0]["text"])


if __name__ == "__main__":
    unittest.main()
