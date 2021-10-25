"""pipeline_mda.py
------------------
Batch pipeline that walks through pre-downloaded 10-K files, extracts the MD&A
section via ``mda_extractor.extract_mda_from_file`` and materialises the result
as JSONL for downstream NLP tasks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Optional

from tqdm import tqdm

from .mda_extractor import extract_mda_from_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base folder that contains per-company sub-directories (already in repo).
DATA_ROOT = Path("Scrape_Data_New/10_k")

# Choose a handful of companies (hand-picked CIK folders)
DEFAULT_COMPANIES = [
    "0000320193",  # Apple
    "0000070858",  # Procter & Gamble
    "0000796343",  # PepsiCo
]

# We look for files in each company folder under *rawtext/* and *grabbed_text/*.
TARGET_SUBDIRS = ["rawtext", "grabbed_text"]

# Output file (JSON Lines)
OUTPUT_PATH = Path("data/processed/mda_sections.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_target_files(company_dir: Path) -> Iterable[Path]:
    for sub in TARGET_SUBDIRS:
        path = company_dir / sub
        if not path.exists():
            continue
        for file in path.glob("*.txt"):
            yield file

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(companies: Optional[List[str]] = None) -> List[Dict]:
    companies = companies or DEFAULT_COMPANIES
    results: List[Dict] = []

    for company_cik in companies:
        company_dir = DATA_ROOT / company_cik
        if not company_dir.exists():
            print(f"[warn] company folder not found: {company_cik}")
            continue

        for file_path in tqdm(list(_iter_target_files(company_dir)), desc=company_cik):
            try:
                mda_text = extract_mda_from_file(file_path)
            except Exception as exc:
                print(f"[error] {file_path}: {exc}")
                continue

            if not mda_text:
                continue  # Skip if MD&A not found

            # Derive filing date from filename pattern 0000320193_2002-12-19.txt
            filing_date = file_path.stem.split("_")[-1]
            results.append(
                {
                    "company_cik": company_cik,
                    "filing_date": filing_date,
                    "source_file": str(file_path),
                    "mda": mda_text,
                }
            )

    # Write JSONL
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        for rec in results:
            json.dump(rec, fp, ensure_ascii=False)
            fp.write("\n")

    print(f"[done] wrote {len(results)} MD&A records to {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    run_pipeline() 