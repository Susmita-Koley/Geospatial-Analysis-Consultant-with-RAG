"""
config.py — centralised settings loaded from .env
All other modules import from here; never import os.getenv() directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of cwd)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


# ── Groq / LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", str(_ROOT / "docs")))
CHROMA_PERSIST_DIR: Path = Path(
    os.getenv("CHROMA_PERSIST_DIR", str(_ROOT / "data" / "chroma_db"))
)
TESTSET_PATH: Path = Path(
    os.getenv("TESTSET_PATH", str(_ROOT / "data" / "testset.json"))
)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION: str = "geospatial_docs"

# ── Validation ────────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "1. Copy .env.example -> .env\n"
        "2. Add your key from https://console.groq.com\n"
    )
