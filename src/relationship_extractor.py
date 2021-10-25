"""relationship_extractor.py
---------------------------
Light-weight helper to extract (subject, verb, object) relationship triples from
a sentence using spaCy’s dependency parse. It is **heuristic** – sufficient for
our prototype – but easy to refine later.
"""

from __future__ import annotations

from typing import List, Tuple

import spacy
from spacy.tokens import Doc, Token
from spacy.symbols import nsubj, nsubjpass, dobj, pobj, VERB

# Attempt to load a small English model (installed during previous steps)
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    # Should already be installed; otherwise raise a clear error.
    raise RuntimeError(
        "spaCy language model 'en_core_web_sm' not found – please run 'python -m spacy download en_core_web_sm'"
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _is_valid_token(tok: Token) -> bool:
    """Filter out pronouns and punctuation tokens."""
    return not (tok.is_stop or tok.is_punct or tok.pos_ == "PRON")


def extract_svo_triples(text: str, *, merge_phrases: bool = True) -> List[Tuple[str, str, str]]:
    """Return list of (subject, verb, object) triples extracted from *text*.

    • Works sentence-by-sentence; multiple triples may come from one sentence.
    • If *merge_phrases* is True, contiguous tokens that form a noun-phrase are
      concatenated into a single string.
    """
    triples: List[Tuple[str, str, str]] = []

    doc: Doc = _NLP(text)

    for sent in doc.sents:
        # Build mapping from token to its potential verb
        verbs = [tok for tok in sent if tok.pos == VERB]
        if not verbs:
            continue

        # For each verb, look for nominal subject and direct / prepositional object
        for verb in verbs:
            subj_tokens = [tok for tok in verb.lefts if tok.dep in (nsubj, nsubjpass)]
            if not subj_tokens:
                continue
            obj_tokens = [tok for tok in verb.rights if tok.dep in (dobj, pobj)]
            if not obj_tokens:
                # try object via prepositional chain e.g., "invested in company"
                for pp in verb.rights:
                    if pp.dep_ == "prep":
                        obj_tokens = [child for child in pp.rights if child.dep in (pobj, dobj)]
                        if obj_tokens:
                            break
            if not obj_tokens:
                continue

            # Build strings
            subj = _span_text(subj_tokens, merge_phrases)
            obj = _span_text(obj_tokens, merge_phrases)
            if subj and obj:
                triples.append((subj, verb.lemma_.lower(), obj))

    return triples


def _span_text(tokens: List[Token], merge_phrases: bool) -> str:
    if not tokens:
        return ""
    if merge_phrases:
        span = tokens[0].doc[tokens[0].left_edge.i : tokens[-1].right_edge.i + 1]
    else:
        span = tokens[0].doc[tokens[0].i : tokens[-1].i + 1]
    # Clean text
    text = " ".join(tok.text for tok in span if _is_valid_token(tok))
    return text.strip()


if __name__ == "__main__":
    sample = "Apple increased its dividend and repurchased shares during the quarter."
    print(extract_svo_triples(sample)) 