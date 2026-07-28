"""
download_corpus.py — Download all 7 geospatial PDF documents for Project 1

Run once before building the vector store:
    python scripts/download_corpus.py

All documents are freely available from official ESA, NASA, and USGS sources.
No login or API key required.
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOCS_DIR = ROOT / "docs"

# ---------------------------------------------------------------------------
# Document registry — official public sources
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "filename": "Sentinel-2_User_Handbook.pdf",
        "url": "https://sentinel.esa.int/documents/247904/685211/Sentinel-2_User_Handbook",
        "size_mb": 1.7,
        "description": "Sentinel-2 MSI User Handbook (ESA)",
    },
    {
        "filename": "S2-PDGS-CS-DI-PSD-V15.0.pdf",
        "url": "https://sentinel.esa.int/documents/247904/349490/S2_MSI_Product_Specification.pdf",
        "size_mb": 8.7,
        "description": "Sentinel-2 Product Specification v15.0 (ESA)",
    },
    {
        "filename": "MOD11_User_Guide_V61.pdf",
        "url": "https://lpdaac.usgs.gov/documents/715/MOD11_User_Guide_V61.pdf",
        "size_mb": 1.0,
        "description": "MODIS LST User Guide C6.1 (MOD11, LP DAAC)",
    },
    {
        "filename": "MCD12_User_Guide_V61.pdf",
        "url": "https://lpdaac.usgs.gov/documents/101/MCD12_User_Guide_V6.pdf",
        "size_mb": 0.4,
        "description": "MODIS Land Cover User Guide C6.1 (MCD12, LP DAAC)",
    },
    {
        "filename": "MOD13_User_Guide_V61.pdf",
        "url": "https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf",
        "size_mb": 2.4,
        "description": "MODIS Vegetation Index User Guide C6.1 (MOD13, LP DAAC)",
    },
    {
        "filename": "MOD17C61UsersGuideV10Feb2021.pdf",
        "url": "https://lpdaac.usgs.gov/documents/972/MOD17_User_Guide_V61.pdf",
        "size_mb": 24.0,
        "description": "MODIS NPP/GPP User Guide C6.1 (MOD17, LP DAAC/NASA)",
    },
    {
        "filename": "LSDS-1822_Landsat8-9-OLI-TIRS_C2_L1_DataFormatControlBook-v7.pdf",
        "url": (
            "https://www.usgs.gov/media/files/"
            "landsat-collection-2-level-1-data-format-control-book"
        ),
        "size_mb": 0.6,
        "description": "Landsat Collection 2 Data Format Control Book (USGS)",
    },
]


def download_file(url: str, dest: Path, description: str, timeout: int = 60) -> bool:
    """Download a single file with progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=description[:45], leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True

    except requests.RequestException as e:
        print(f"  [ERROR] {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(DOCUMENTS)} documents to: {DOCS_DIR}\n")

    ok, skipped, failed = 0, 0, []

    for doc in DOCUMENTS:
        dest = DOCS_DIR / doc["filename"]
        print(f"  {doc['description']}")

        if dest.exists():
            size_kb = dest.stat().st_size // 1024
            print(f"  [SKIP] Already exists ({size_kb} KB)")
            skipped += 1
            continue

        success = download_file(doc["url"], dest, doc["description"])
        if success:
            size_kb = dest.stat().st_size // 1024
            print(f"  [OK] Downloaded ({size_kb} KB)")
            ok += 1
        else:
            print(f"  [FAIL] Manual download required — see README for direct link")
            failed.append(doc["filename"])
        print()

    print("=" * 55)
    print(f"Done: {ok} downloaded, {skipped} already present, {len(failed)} failed")

    if failed:
        print("\nFailed downloads (download manually and place in docs/):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll documents ready. Next step:")
        print("  python -m src.rag.ingest")


if __name__ == "__main__":
    main()
