"""
ingest.py — PDF ingestion pipeline
  1. Parse PDFs with PyMuPDF (page-by-page, preserves page numbers for citations)
  2. Split into chunks with RecursiveCharacterTextSplitter
  3. Embed with SentenceTransformers (all-MiniLM-L6-v2, runs locally)
  4. Persist to ChromaDB

Run standalone:  python -m src.rag.ingest
Or import:        from src.rag.ingest import build_vectorstore
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── PDF Parser ────────────────────────────────────────────────────────────────

def parse_pdfs(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """
    Extract text from every PDF in docs_dir.
    Each page becomes a LangChain Document with rich metadata:
      - source: filename
      - page: 1-indexed page number
      - total_pages: total pages in the PDF
    """
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")

    log.info("Found %d PDFs in %s", len(pdf_files), docs_dir)
    documents: list[Document] = []

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:          # skip blank/image-only pages
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "total_pages": total_pages,
                    },
                )
            )
        doc.close()

    log.info("Extracted %d pages total", len(documents))
    return documents


# ── Splitter ──────────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split pages into overlapping chunks.
    Metadata (source, page) is inherited by every child chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ── Embedding & Vector Store ──────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embeddings object (downloaded once, then local)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Build (or load) the ChromaDB vector store.

    Args:
        force_rebuild: If True, delete existing store and re-embed everything.

    Returns:
        A Chroma retriever-ready vector store.
    """
    persist_dir = str(CHROMA_PERSIST_DIR)

    if not force_rebuild and CHROMA_PERSIST_DIR.exists():
        log.info("Loading existing ChromaDB from %s", persist_dir)
        embeddings = get_embeddings()
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    log.info("Building new ChromaDB vector store …")
    documents = parse_pdfs()
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=persist_dir,
    )
    log.info("ChromaDB saved to %s (%d chunks indexed)", persist_dir, len(chunks))
    return vectorstore


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument(
        "--force", action="store_true", help="Re-embed even if store already exists"
    )
    args = parser.parse_args()
    build_vectorstore(force_rebuild=args.force)
    print("✅ Ingestion complete. Run the Streamlit app next: streamlit run app.py")
