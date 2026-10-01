"""
RAG Assistant Module
====================
Retrieves relevant document chunks from the vector store, constructs context-augmented
prompts, calls LLM (OpenAI API if key available, or offline template generator),
and returns answers along with detailed source citations.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.vector_store import VectorStore

# Load environment variables from .env file
load_dotenv()


class RAGAssistant:
    """
    RAG Assistant for answering questions using ingested document knowledge base.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        # Attempt to load existing index if available
        self.vector_store.load()

    def format_citation(self, metadata: Dict[str, Any]) -> str:
        """
        Formats metadata into readable source citation string.
        """
        source = metadata.get("source", "Unknown Document")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id", 0)

        page_str = f", Page {page}" if page is not None else ""
        return f"[Source: {source}{page_str}, Chunk #{chunk_id}]"

    def generate_llm_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates answer using OpenAI API if key available, otherwise falls back to structured context answer.
        """
        api_key = os.getenv("OPENAI_API_KEY")

        context_str = ""
        citations = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            cite = self.format_citation(chunk["metadata"])
            citations.append(f"{i}. {cite}\n\"{chunk['text']}\"")

        context_str = "\n\n".join(citations)

        if api_key and api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                system_prompt = (
                    "You are a helpful AI Knowledge Assistant. Answer the user's question accurately "
                    "using ONLY the provided source context chunks. Always include citations in your "
                    "answer using the format [Source: <filename>, Page <page>, Chunk #<id>]. "
                    "If the answer cannot be found in the context, state clearly that the provided "
                    "documents do not contain the answer."
                )

                user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                # Fall back if OpenAI API call fails or encounters network/auth error
                pass

        # Offline / Fallback answer generation
        if not retrieved_chunks:
            return "No relevant documents found in the vector store to answer your question."

        fallback_msg = (
            "Based on retrieved document context:\n\n"
        )
        for chunk in retrieved_chunks:
            cite = self.format_citation(chunk["metadata"])
            fallback_msg += f"• **Excerpt ({cite})**:\n\"{chunk['text']}\"\n\n"

        return fallback_msg.strip()

    def answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves top_k chunks for query and returns answer dict with context and sources.

        Returns:
            Dict: {
                "query": str,
                "answer": str,
                "sources": List[Dict[str, Any]]
            }
        """
        results = self.vector_store.search(query, top_k=top_k)
        retrieved_chunks = [item[0] for item in results]

        answer = self.generate_llm_answer(query, retrieved_chunks)

        sources = []
        for chunk in retrieved_chunks:
            meta = chunk["metadata"]
            sources.append({
                "citation": self.format_citation(meta),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            })

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }
