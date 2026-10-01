import unittest

from src.vector_store import VectorStore
from src.rag_assistant import RAGAssistant


class TestRAGAssistant(unittest.TestCase):

    def setUp(self):
        self.vector_store = VectorStore()
        self.sample_chunks = [
            {
                "text": "Supervised learning uses labeled training datasets to train models.",
                "metadata": {"source": "ai_concepts.pdf", "page": 3, "chunk_id": 2}
            },
            {
                "text": "Unsupervised learning discovers hidden patterns or data groupings without labeled responses.",
                "metadata": {"source": "ai_concepts.pdf", "page": 4, "chunk_id": 3}
            }
        ]
        self.vector_store.add_documents(self.sample_chunks)
        self.assistant = RAGAssistant(vector_store=self.vector_store)

    def test_format_citation(self):
        meta = {"source": "doc.pdf", "page": 5, "chunk_id": 1}
        citation = self.assistant.format_citation(meta)
        self.assertEqual(citation, "[Source: doc.pdf, Page 5, Chunk #1]")

    def test_answer_generation(self):
        result = self.assistant.answer("What is supervised learning?", top_k=1)

        self.assertIn("query", result)
        self.assertIn("answer", result)
        self.assertIn("sources", result)

        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("ai_concepts.pdf", result["sources"][0]["citation"])
        self.assertIn("Supervised learning", result["answer"])


if __name__ == "__main__":
    unittest.main()
