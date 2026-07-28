"""
generate_testset.py — Auto-generate QA pairs using RAGAS 0.4.x TestsetGenerator

USAGE:
    python scripts/generate_testset.py

OUTPUT:
    data/testset.json  (60 QA pairs)

WHY: Manual QA annotation is biased and time-consuming.
     RAGAS TestsetGenerator creates diverse, realistic questions
     directly from your documents using LLM-driven evolution strategies.
"""

import logging
import sys
from pathlib import Path

# Add project root to path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def main():
    TESTSET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TESTSET_PATH.exists():
        overwrite = input(f"Testset already exists at {TESTSET_PATH}. Overwrite? [y/N]: ")
        if overwrite.strip().lower() != "y":
            print("Aborted.")
            return

    # ── Load documents ────────────────────────────────────────────────────────
    log.info("Loading PDFs from %s …", DOCS_DIR)
    documents = parse_pdfs(DOCS_DIR)
    log.info("Loaded %d pages", len(documents))

    # ── LLM + embeddings wrapped for RAGAS 0.4.x ────────────────────────────
    generator_llm = LangchainLLMWrapper(
        ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.3)
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    # ── RAGAS 0.4.x TestsetGenerator ─────────────────────────────────────────
    # from_langchain takes wrapped LLM and embeddings
    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings,
    )

    log.info("Generating testset (this may take 10–20 minutes) …")
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=60,
    )

    # ── Save to JSON ──────────────────────────────────────────────────────────
    df = testset.to_pandas()
    df.to_json(str(TESTSET_PATH), orient="records", indent=2)
    log.info("Saved %d QA pairs to %s", len(df), TESTSET_PATH)

    # Print sample
    print("\n=== Sample Generated Questions ===")
    for _, row in df.head(3).iterrows():
        q = row.get("user_input") or row.get("question", "")
        a = str(row.get("reference") or row.get("ground_truth", ""))
        print(f"\nQ: {q}")
        print(f"A: {a[:200]}…")

    print(f"\n✅ Testset generation complete: {len(df)} QA pairs → {TESTSET_PATH}")
    print("Next: python -m src.rag.evaluator")


if __name__ == "__main__":
    main()
