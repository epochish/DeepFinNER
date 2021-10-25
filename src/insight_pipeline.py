"""insight_pipeline.py
---------------------
Generate analyst-friendly insights over previously extracted MD&A sections.
For each MD&A record we run:
  • Named-entity extraction via `FinancialEntityExtractor` (pattern + spaCy).
  • Sentence-level sentiment using FinBERT (3-class: positive / negative / neutral).
  • Aggregate sentiment scores.
Results are stored to ``data/processed/mda_insights.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm
from transformers import pipeline as hf_pipeline, AutoTokenizer

from .entity_extractor import FinancialEntityExtractor
from .nlp_processor import NLPProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = Path("data/processed/mda_sections.jsonl")
OUTPUT_PATH = Path("data/processed/mda_insights.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SENTIMENT_MODEL_NAME = "yiyanghkust/finbert-tone"
SUMMARY_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_records() -> List[Dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    with INPUT_PATH.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def _save_records(records: List[Dict]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        for rec in records:
            json.dump(rec, fp, ensure_ascii=False)
            fp.write("\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(force: bool = False) -> List[Dict]:
    if OUTPUT_PATH.exists() and not force:
        print(f"[skip] Insights already exist at {OUTPUT_PATH}. Use force=True to regenerate.")
        with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
            return [json.loads(line) for line in fp]

    print("[info] Loading FinBERT sentiment model – this may take a minute on first run…")
    sentiment_pipe = hf_pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL_NAME,
        tokenizer=SENTIMENT_MODEL_NAME,
        truncation=True,
        max_length=256,  # conservative limit to avoid positional embedding overflow
    )

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)

    print("[info] Loading summarization model …")
    summary_pipe = hf_pipeline("summarization", model=SUMMARY_MODEL_NAME, tokenizer=SUMMARY_MODEL_NAME)

    extractor = FinancialEntityExtractor()
    nlp_proc = NLPProcessor()

    input_records = _load_records()
    output_records: List[Dict] = []

    for rec in tqdm(input_records, desc="generating-insights"):
        text: str = rec["mda"]

        # Sentence segmentation (NLTK already downloaded in NLPProcessor)
        sentences = nlp_proc.tokenize_text(text, method="sentence")

        # Filter and chunk overly long sentences to avoid 512-token overflow.
        processed_results = []
        for sent in sentences:
            # Determine token length quickly; fallback split if required
            token_count = len(tokenizer.encode(sent, add_special_tokens=False))
            if token_count <= 512:
                processed_results.extend(sentiment_pipe(sent, max_length=512, truncation=True))
            else:
                # Rough split by words into ~256-token chunks
                words = sent.split()
                chunk_size = 256
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i : i + chunk_size])
                    processed_results.extend(
                        sentiment_pipe(chunk, max_length=512, truncation=True)
                    )

        sent_results = processed_results

        # Aggregate sentiment counts
        sentiment_counts: Dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for sr in sent_results:
            label = sr["label"].lower()
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

        total_sents = sum(sentiment_counts.values()) or 1
        sentiment_scores = {k: v / total_sents for k, v in sentiment_counts.items()}

        # NER extraction
        entities = extractor.comprehensive_entity_extraction(text)

        # Executive summary (truncate to 1024 tokens for speed)
        summary_input = text[:4000]  # rough char limit
        summary = summary_pipe(summary_input, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]

        # Assemble insight object
        output_records.append(
            {
                **rec,
                "summary": summary,
                "sentiment": {
                    "sentence_breakdown": sent_results,  # raw list for transparency
                    "aggregate": sentiment_scores,
                },
                "entities": entities,
            }
        )

    _save_records(output_records)
    print(f"[done] wrote {len(output_records)} insight records to {OUTPUT_PATH}")
    return output_records


if __name__ == "__main__":
    run_pipeline() 