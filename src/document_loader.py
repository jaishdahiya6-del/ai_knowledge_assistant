"""
Document Loader and Text Splitter Module
=======================================
Supports loading PDF, TXT, and Markdown files, extracting text along with
metadata, and splitting text into overlapping chunks for vector embedding.
"""

import os
from typing import List, Dict, Any, Optional
from pypdf import PdfReader


def load_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Reads a single file (.pdf, .txt, .md) and returns a list of page/document records
    with content and metadata.

    Returns:
        List of dicts: [{"text": str, "metadata": {"source": str, "page": Optional[int]}}]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    results = []

    if ext == ".pdf":
        reader = PdfReader(filepath)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                results.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": filename,
                        "filepath": filepath,
                        "page": idx + 1
                    }
                })
    elif ext in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            if text.strip():
                results.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": filename,
                        "filepath": filepath,
                        "page": 1
                    }
                })
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .txt, .md")

    return results


def load_directory(dirpath: str) -> List[Dict[str, Any]]:
    """
    Loads all supported files (.pdf, .txt, .md) from a directory recursively.
    """
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f"Directory not found: {dirpath}")

    documents = []
    supported_extensions = {".pdf", ".txt", ".md"}

    for root, _, files in os.walk(dirpath):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                filepath = os.path.join(root, file)
                try:
                    docs = load_file(filepath)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")

    return documents


def chunk_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping chunks of approximately `chunk_size` characters,
    preserving metadata and assigning chunk IDs.

    Args:
        text: Plain text content to split.
        metadata: Base metadata dictionary to attach to each chunk.
        chunk_size: Target character length for each chunk.
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of chunk dicts: [{"text": str, "metadata": Dict[str, Any]}]
    """
    if not text or chunk_size <= 0:
        return []

    if metadata is None:
        metadata = {}

    chunks = []
    start = 0
    text_len = len(text)
    chunk_idx = 0

    while start < text_len:
        end = start + chunk_size
        chunk_str = text[start:end]

        # Try to break cleanly at sentence or line boundaries if possible
        if end < text_len:
            last_space = max(chunk_str.rfind("\n"), chunk_str.rfind(". "), chunk_str.rfind(" "))
            if last_space > chunk_size // 2:
                end = start + last_space + 1
                chunk_str = text[start:end]

        chunk_meta = dict(metadata)
        chunk_meta["chunk_id"] = chunk_idx

        chunks.append({
            "text": chunk_str.strip(),
            "metadata": chunk_meta
        })

        chunk_idx += 1
        start = end - chunk_overlap if (end - chunk_overlap) > start else end

    return chunks


def process_file_or_directory(path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Convenience function to load file(s) from a path (file or dir) and split them into chunks.
    """
    if os.path.isfile(path):
        docs = load_file(path)
    elif os.path.isdir(path):
        docs = load_directory(path)
    else:
        raise ValueError(f"Invalid path: {path}")

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(
            text=doc["text"],
            metadata=doc["metadata"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        all_chunks.extend(chunks)

    return all_chunks
