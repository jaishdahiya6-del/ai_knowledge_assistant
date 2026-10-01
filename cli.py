#!/usr/bin/env python3
"""
CLI Tool for AI Knowledge Assistant
===================================
Command-line interface to ingest documents (.pdf, .txt, .md) and query the vector store
for answers with source citations.

Usage:
    python cli.py ingest --path docs/
    python cli.py ask "What is gradient descent?"
"""

import sys
import os
import argparse
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.document_loader import process_file_or_directory
from src.vector_store import VectorStore
from src.rag_assistant import RAGAssistant

load_dotenv()


def handle_ingest(args):
    path = args.path
    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)

    print(f"Ingesting documents from '{path}'...")
    chunks = process_file_or_directory(
        path=path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    if not chunks:
        print("No valid document chunks found to ingest.")
        return

    print(f"Processed {len(chunks)} text chunk(s). Updating vector store...")
    vector_store = VectorStore()
    # Load existing if available to append
    vector_store.load(args.save_dir)
    vector_store.add_documents(chunks)
    vector_store.save(args.save_dir)

    print(f"Successfully ingested and indexed {len(chunks)} chunk(s) into '{args.save_dir}'.")


def handle_ask(args):
    question = args.question
    vector_store = VectorStore()
    if not vector_store.load(args.save_dir):
        print(f"Error: No vector store found at '{args.save_dir}'. Please ingest documents first using 'python cli.py ingest --path <path>'.")
        sys.exit(1)

    assistant = RAGAssistant(vector_store=vector_store)
    print(f"\nQuestion: {question}\n")
    print("Retrieving context and generating answer...\n")

    res = assistant.answer(question, top_k=args.top_k)

    print("=== Answer ===")
    print(res["answer"])
    print("\n=== Citations & Sources ===")
    if not res["sources"]:
        print("No sources found.")
    else:
        for idx, src in enumerate(res["sources"], 1):
            print(f"{idx}. {src['citation']}")
            print(f"   Snippet: {src['snippet']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI Knowledge Assistant CLI - Document Ingestion & RAG QA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF, TXT, or MD documents into vector store")
    ingest_parser.add_argument("--path", "-p", required=True, help="Path to document file or directory")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters (default: 500)")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in characters (default: 50)")
    ingest_parser.add_argument("--save-dir", default="data/vector_store", help="Vector store save directory (default: data/vector_store)")
    ingest_parser.set_defaults(func=handle_ingest)

    # Subcommand: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question to the knowledge assistant")
    ask_parser.add_argument("question", type=str, help="Question to ask the assistant")
    ask_parser.add_argument("--top-k", type=int, default=3, help="Number of context chunks to retrieve (default: 3)")
    ask_parser.add_argument("--save-dir", default="data/vector_store", help="Vector store load directory (default: data/vector_store)")
    ask_parser.set_defaults(func=handle_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
