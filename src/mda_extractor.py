"""
MD&A Extractor
--------------
Utility functions that locate and extract the *Management’s Discussion and Analysis* (MD&A)
section from SEC 10-K filings. This is the first step in our streamlined prototype pipeline.

The logic is intentionally heuristic-based – good enough for a proof-of-concept – and keeps
external dependencies to a minimum.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Regex patterns – broadened to catch more formatting variants
# ---------------------------------------------------------------------------

# Start looks like: "Item 7. Management’s Discussion and Analysis", but
# sometimes the dot is a dash, colon, or absent. Allow up to 10 non-alpha chars
# between "7" and "Management" to account for these quirks.
_START_REGEX = re.compile(
    r"item\s+7[^a-zA-Z]{0,10}management[\s\S]{0,120}?discussion[\s\S]{0,120}?analysis",
    flags=re.IGNORECASE,
)

# End when Item 7A (market-risk disclosures) *or* Item 8 (Financial Statements)
# appears. Allow optional punctuation/dash afterwards.
_END_REGEX = re.compile(r"item\s+7a\s*[\.:\-]|item\s+8\s*[\.:\-]", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_mda_from_text(text: str) -> Optional[str]:
    """Return the MD&A section from *text* or *None* if not found.

    The function looks for a start marker matching ``item 7. management … analysis`` and takes
    everything until the next section marker (``item 7a`` or ``item 8``).
    """
    # Normalise whitespace for easier regex matching (but keep original indices)
    normalised = re.sub(r"\s+", " ", text)

    start_match = _START_REGEX.search(normalised)
    if not start_match:
        return None

    end_match = _END_REGEX.search(normalised, pos=start_match.end())
    end_pos = end_match.start() if end_match else len(normalised)

    return normalised[start_match.start(): end_pos].strip()


def extract_mda_from_file(path: Union[str, Path]) -> Optional[str]:
    """Read *path* (HTML or plain-text) and return the extracted MD&A section."""
    path = Path(path)
    data = path.read_text(encoding="utf-8", errors="ignore")

    # Quick heuristic: try HTML parsing if looks like HTML, else treat as plain text.
    if "<html" in data.lower() or "</" in data[:400].lower():
        soup = BeautifulSoup(data, "lxml")
        data = soup.get_text(" ")

    return extract_mda_from_text(data)


# ---------------------------------------------------------------------------
# CLI helper for ad-hoc testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="Extract MD&A from a 10-K document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              python -m src.mda_extractor path/to/10k_file.html > mda.txt
            """,
        ),
    )
    parser.add_argument("file", type=str, help="Path to 10-K HTML or text file")
    args = parser.parse_args()

    section = extract_mda_from_file(args.file)
    if section is None:
        print("[MD&A section not found]", flush=True)
    else:
        print(section, flush=True) 