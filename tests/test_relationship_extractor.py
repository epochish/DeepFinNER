import pytest

from src.relationship_extractor import extract_svo_triples


def test_simple_svo():
    text = "Apple acquired Beats."
    triples = extract_svo_triples(text)
    assert ("Apple", "acquire", "Beats") in triples 