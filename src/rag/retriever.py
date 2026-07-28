"""
retriever.py — RAG query pipeline
  1. Load ChromaDB vector store
  2. Retrieve top-k relevant chunks
  3. Build a prompt with retrieved context
  4. Stream answer from Groq API
  5. Return answer + source citations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K,
)
from src.rag.ingest import get_embeddings

log = logging.getLogger(__name__)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Citation:
    source: str
    page: int

@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a geospatial document assistant with expertise in satellite remote sensing.
Answer questions strictly based on the provided context excerpts from official ESA, NASA, and USGS documentation.

Rules:
1. Answer ONLY from the provided context. Do not hallucinate facts.
2. If the context does not contain enough information, say "I don't have enough context to answer this accurately."
3. Cite your sources by referencing the document name and page number.
4. Be precise — satellite specifications, band numbers, wavelengths, and calibration values must be exact.
5. Structure long answers with bullet points or numbered lists for clarity."""


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Format the RAG prompt with retrieved context."""
    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return f"""CONTEXT (from official documentation):
{context_text}

QUESTION: {question}

ANSWER (cite document + page for each fact):"""


# ── Core retriever ────────────────────────────────────────────────────────────

class GeospatialRAG:
    """
    Stateless RAG query handler.
    Loads the persistent ChromaDB once per session; all queries reuse it.
    """

    def __init__(self):
        self._vectorstore: Chroma | None = None
        self._llm: ChatGroq | None = None

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            persist_dir = str(CHROMA_PERSIST_DIR)
            if not CHROMA_PERSIST_DIR.exists():
                raise RuntimeError(
                    "ChromaDB not found. Run ingestion first:\n"
                    "  python -m src.rag.ingest\n"
                    "Or check the Setup tab in the app."
                )
            self._vectorstore = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=get_embeddings(),
                persist_directory=persist_dir,
            )
        return self._vectorstore

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL,
                temperature=0.1,       # low temp → factual, consistent answers
                max_tokens=1024,
            )
        return self._llm

    def query(self, question: str, top_k: int = TOP_K) -> RAGResponse:
        """
        Full RAG pipeline: retrieve → prompt → generate → return with citations.

        Args:
            question: Natural language question from the user
            top_k: Number of document chunks to retrieve

        Returns:
            RAGResponse with answer text and source citations
        """
        vs = self._get_vectorstore()
        llm = self._get_llm()

        # ── Retrieval ──────────────────────────────────────────────────────────
        docs_with_scores = vs.similarity_search_with_score(question, k=top_k)

        context_chunks = [d.page_content for d, _ in docs_with_scores]
        citations = [
            Citation(
                source=d.metadata.get("source", "unknown"),
                page=d.metadata.get("page", 0),
            )
            for d, _ in docs_with_scores
        ]

        # Deduplicate citations (same source+page may appear from overlapping chunks)
        seen = set()
        unique_citations: list[Citation] = []
        for c in citations:
            key = (c.source, c.page)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)

        # ── Generation ────────────────────────────────────────────────────────
        prompt = build_prompt(question, context_chunks)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        answer = response.content.strip()

        return RAGResponse(
            answer=answer,
            citations=unique_citations,
            context_chunks=context_chunks,
        )
