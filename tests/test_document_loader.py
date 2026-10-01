import os
import tempfile
import unittest

from src.document_loader import load_file, chunk_text, process_file_or_directory


class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.txt_file = os.path.join(self.tmp_dir.name, "sample.txt")
        self.md_file = os.path.join(self.tmp_dir.name, "sample.md")

        with open(self.txt_file, "w", encoding="utf-8") as f:
            f.write("This is a sample text file for testing document ingestion and chunking.")

        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write("# Sample Header\nMarkdown content line 1.\nMarkdown content line 2.")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_txt_file(self):
        docs = load_file(self.txt_file)
        self.assertEqual(len(docs), 1)
        self.assertIn("sample text file", docs[0]["text"])
        self.assertEqual(docs[0]["metadata"]["source"], "sample.txt")

    def test_load_md_file(self):
        docs = load_file(self.md_file)
        self.assertEqual(len(docs), 1)
        self.assertIn("Sample Header", docs[0]["text"])
        self.assertEqual(docs[0]["metadata"]["source"], "sample.md")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_file("non_existent_file.txt")

    def test_chunk_text(self):
        text = "Paragraph one is short. " * 10
        metadata = {"source": "test.txt", "page": 1}
        chunks = chunk_text(text, metadata, chunk_size=100, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)
            self.assertEqual(chunk["metadata"]["source"], "test.txt")
            self.assertIn("chunk_id", chunk["metadata"])

    def test_process_file_or_directory(self):
        chunks = process_file_or_directory(self.tmp_dir.name, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
