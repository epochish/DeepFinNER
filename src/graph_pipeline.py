"""graph_pipeline.py
--------------------
Construct a directed relationship graph from MD&A insight records.
Nodes: distinct entities (subjects/objects) extracted via SVO triples.
Edges: subject -> object, attributes = {verb, sentiment}.
Graphs are persisted in NetworkX node-link JSON format to
  data/processed/graphs/{company_cik}_{year}.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm

from .relationship_extractor import extract_svo_triples
from .nlp_processor import NLPProcessor

INSIGHT_FILE = Path("data/processed/mda_insights.jsonl")
GRAPH_DIR = Path("data/processed/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def _load_insights() -> List[Dict]:
    if not INSIGHT_FILE.exists():
        raise FileNotFoundError(f"Insights file not found: {INSIGHT_FILE}")
    with INSIGHT_FILE.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def _save_graph(cik: str, year: str, G: nx.DiGraph) -> Path:
    path = GRAPH_DIR / f"{cik}_{year}.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(json_graph.node_link_data(G), fp, ensure_ascii=False)
    return path


def build_graph_for_record(rec: Dict, nlp_proc: NLPProcessor) -> Path:
    cik = rec["company_cik"]
    year = rec["filing_date"].split("-")[0]

    sentences = nlp_proc.tokenize_text(rec["mda"], method="sentence")
    sentiments = rec["sentiment"]["sentence_breakdown"]

    G = nx.DiGraph(company_cik=cik, year=year)

    for idx, sent in enumerate(sentences):
        triples = extract_svo_triples(sent)
        if not triples:
            continue
        sent_label = sentiments[idx]["label"].lower() if idx < len(sentiments) else "unknown"

        for subj, verb, obj in triples:
            if not subj or not obj:
                continue
            # Add / update edge with verb list & sentiment tally
            if G.has_edge(subj, obj):
                data = G.edges[subj, obj]
                verbs = set(data.get("verbs", []))
                verbs.add(verb)
                data["verbs"] = list(verbs)
                # Update sentiment counts
                data[f"sent_{sent_label}"] = data.get(f"sent_{sent_label}", 0) + 1
            else:
                G.add_edge(
                    subj,
                    obj,
                    verbs=[verb],
                    sent_positive=1 if sent_label == "positive" else 0,
                    sent_negative=1 if sent_label == "negative" else 0,
                    sent_neutral=1 if sent_label == "neutral" else 0,
                )

    return _save_graph(cik, year, G)


def run_pipeline() -> None:
    nlp_proc = NLPProcessor()
    insights = _load_insights()
    _paths: List[Path] = []
    for rec in tqdm(insights, desc="build-graphs"):
        path = build_graph_for_record(rec, nlp_proc)
        _paths.append(path)
    print(f"[done] generated {len(_paths)} graph files in {GRAPH_DIR}")


if __name__ == "__main__":
    run_pipeline() 