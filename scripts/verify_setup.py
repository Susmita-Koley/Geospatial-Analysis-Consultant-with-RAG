"""
verify_setup.py — Quick sanity-check script
Run AFTER activating venv and setting up .env

Usage:
    python scripts/verify_setup.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(label, ok, detail=""):
    icon = "[OK]" if ok else "[FAIL]"
    print(f"  {icon} {label}" + (f"  -- {detail}" if detail else ""))
    return ok


print("\n=== Environment Verification - Project 1: Geospatial RAG ===\n")

all_ok = True

# 1. Python version
import platform
py_ver = platform.python_version()
ok = any(py_ver.startswith(v) for v in ("3.12", "3.11", "3.10"))
all_ok &= check(f"Python version: {py_ver}", ok, "3.10+ required")

# 2. .env file
env_path = ROOT / ".env"
all_ok &= check(".env file exists", env_path.exists(), str(env_path))

# 3. GROQ_API_KEY
try:
    from src.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, DOCS_DIR
    key_ok = bool(GROQ_API_KEY) and GROQ_API_KEY != "your_groq_api_key_here"
    all_ok &= check("GROQ_API_KEY configured", key_ok, "Set in .env")
    check(f"Model: {GROQ_MODEL}", True)
    check(f"Embedding: {EMBEDDING_MODEL}", True)
except EnvironmentError as e:
    all_ok &= check("Config loaded", False, str(e)[:80])

# 4. PDF docs
docs_dir = ROOT / "docs"
pdfs = list(docs_dir.glob("*.pdf")) if docs_dir.exists() else []
all_ok &= check(f"PDF documents ({len(pdfs)}/7 found)", len(pdfs) == 7, str(docs_dir))
if pdfs:
    for pdf in pdfs:
        print(f"     [PDF] {pdf.name} ({pdf.stat().st_size // 1024} KB)")

# 5. Key packages
packages = [
    ("langchain", "langchain"),
    ("chromadb", "chromadb"),
    ("langchain_chroma", "langchain-chroma"),
    ("sentence_transformers", "sentence-transformers"),
    ("fitz", "pymupdf"),
    ("ragas", "ragas"),
    ("streamlit", "streamlit"),
    ("langchain_groq", "langchain-groq"),
    ("groq", "groq"),
    ("plotly", "plotly"),
]
print("\n  Package availability:")
for mod, pkg in packages:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        check(f"  {pkg} ({ver})", True)
    except ImportError:
        all_ok &= check(f"  {pkg}", False, f"Run: pip install {pkg}")

# 6. ChromaDB store
chroma_dir = ROOT / "data" / "chroma_db"
chroma_built = chroma_dir.exists() and any(chroma_dir.iterdir()) if chroma_dir.exists() else False
check("ChromaDB vector store built", chroma_built,
      "Run: python -m src.rag.ingest" if not chroma_built else str(chroma_dir))

# 7. Testset
testset = ROOT / "data" / "testset.json"
check("RAGAS testset generated", testset.exists(),
      "Run: python scripts/generate_testset.py" if not testset.exists() else str(testset))

# Summary
print()
if all_ok:
    print("[PASS] All checks passed! Run: streamlit run app.py")
else:
    print("[WARN] Some checks failed -- fix the [FAIL] items above, then re-run.")
    print("       Note: ChromaDB and testset are built AFTER installing, so those")
    print("       [FAIL] items are expected on first run.")
print()
