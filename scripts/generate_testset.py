"""
generate_testset.py — Auto-generate QA pairs using RAGAS 0.4.x TestsetGenerator

USAGE:
    python scripts/generate_testset.py

OUTPUT:
    data/testset.json  (~60 QA pairs)

NOTES:
- The script is safe to re-run if interrupted — it will ask before overwriting.
- Network drops are handled with automatic retry (up to 5 attempts, exponential backoff).
- Add HUGGINGFACE_HUB_TOKEN to .env to avoid HuggingFace rate-limits on model downloads.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path so `src` is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Load env vars before any other imports ────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Set HuggingFace token early so transformers/sentence-transformers picks it up
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
if hf_token and hf_token != "your_hf_token_here":
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

from src.config import (
    DOCS_DIR,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    TESTSET_PATH,
)
from src.rag.ingest import parse_pdfs

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # seconds (doubles each attempt: 10, 20, 40, 80, 160)


def generate_with_retry(generator, documents, testset_size):
    """
    Call generator.generate_with_langchain_docs() with automatic retry on
    network errors (WiFi drops, timeouts, connection resets).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("Generation attempt %d/%d …", attempt, MAX_RETRIES)
            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=testset_size,
            )
            return testset

        except Exception as e:
            err_str = str(e).lower()
            is_network = any(
                kw in err_str for kw in [
                    "connection", "timeout", "network", "ssl", "reset",
                    "eof", "broken pipe", "remote host", "10054",
                ]
            )

            if attempt == MAX_RETRIES:
                log.error("All %d attempts failed. Last error: %s", MAX_RETRIES, e)
                raise

            if is_network:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "Network error (attempt %d/%d): %s\n"
                    "  -> Retrying in %d seconds …",
                    attempt, MAX_RETRIES, type(e).__name__, delay
                )
                time.sleep(delay)
            else:
                # Non-network error — don't retry
                log.error("Non-retryable error: %s", e)
                raise


def main():
    TESTSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TESTSET_PATH.exists():
        overwrite = input(f"\nTestset already exists at {TESTSET_PATH}.\nOverwrite? [y/N]: ")
        if overwrite.strip().lower() != "y":
            print("Aborted. Existing testset preserved.")
            return

    # ── Load documents ────────────────────────────────────────────────────────
    log.info("Loading PDFs from %s ...", DOCS_DIR)
    documents = parse_pdfs(DOCS_DIR)
    log.info("Loaded %d pages", len(documents))

    # ── LLM + embeddings ──────────────────────────────────────────────────────
    log.info("Initialising LLM: %s", GROQ_MODEL)
    generator_llm = LangchainLLMWrapper(
        ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.3,
                 max_retries=3)  # Groq client-level retries
    )

    log.info("Initialising embeddings: %s (local CPU)", EMBEDDING_MODEL)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    # ── TestsetGenerator ──────────────────────────────────────────────────────
    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings,
    )

    log.info(
        "Generating testset (60 QA pairs). This takes 10-20 min.\n"
        "  Network drops are handled with auto-retry (up to %d attempts).",
        MAX_RETRIES,
    )

    testset = generate_with_retry(generator, documents, testset_size=60)

    # ── Save ──────────────────────────────────────────────────────────────────
    df = testset.to_pandas()
    df.to_json(str(TESTSET_PATH), orient="records", indent=2)
    log.info("Saved %d QA pairs to %s", len(df), TESTSET_PATH)

    # Print sample
    print("\n=== Sample Generated Questions ===")
    for _, row in df.head(3).iterrows():
        q = row.get("user_input") or row.get("question", "")
        a = str(row.get("reference") or row.get("ground_truth", ""))
        print(f"\nQ: {q}")
        print(f"A: {a[:200]}...")

    print(f"\n[DONE] {len(df)} QA pairs saved to {TESTSET_PATH}")
    print("Next: python -m src.rag.evaluator")


if __name__ == "__main__":
    main()
