from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
from typing import List, Optional

app = FastAPI(title="DeepFinNER MD&A API", version="0.1.0")

DATA_FILE = Path("data/processed/mda_sections.jsonl")


class MDARecord(BaseModel):
    company_cik: str
    filing_date: str  # YYYY-MM-DD
    source_file: str
    mda: str


def _load_records() -> List[MDARecord]:
    if not DATA_FILE.exists():
        return []
    with DATA_FILE.open("r", encoding="utf-8") as fp:
        return [MDARecord(**json.loads(line)) for line in fp]


@app.get("/mda/{company_cik}", response_model=List[MDARecord])
async def get_mda(company_cik: str, year: Optional[int] = None):
    db = [rec for rec in _load_records() if rec.company_cik == company_cik]
    if year is not None:
        db = [rec for rec in db if rec.filing_date.startswith(str(year))]
    if not db:
        raise HTTPException(status_code=404, detail="MD&A not found for given parameters")
    return db


INSIGHT_FILE = Path("data/processed/mda_insights.jsonl")


def _load_insights() -> List[dict]:
    if not INSIGHT_FILE.exists():
        return []
    with INSIGHT_FILE.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


@app.get("/insights/{company_cik}")
async def get_insights(company_cik: str, year: Optional[int] = None):
    """Return enriched NER + sentiment insights for a company (optionally filter by year)."""
    db = [rec for rec in _load_insights() if rec["company_cik"] == company_cik]
    if year is not None:
        db = [rec for rec in db if rec["filing_date"].startswith(str(year))]
    if not db:
        raise HTTPException(status_code=404, detail="Insight not found for given parameters")
    return db


GRAPH_DIR = Path("data/processed/graphs")


@app.get("/graph/{company_cik}")
async def get_graph(company_cik: str, year: Optional[int] = None):
    """Return relationship graph JSON for company/year (defaults to all years combined)."""
    if not GRAPH_DIR.exists():
        raise HTTPException(status_code=404, detail="Graph data not generated yet")

    graphs = []
    for file in GRAPH_DIR.glob("*.json"):
        if not file.stem.startswith(company_cik):
            continue
        if year is not None and not file.stem.endswith(str(year)):
            continue
        graphs.append(json.loads(file.read_text()))

    if not graphs:
        raise HTTPException(status_code=404, detail="No graph found for given parameters")

    # If multiple years requested, return list; else single
    return graphs if year is None else graphs[0]


@app.get("/health")
async def health():
    return {"status": "ok"} 