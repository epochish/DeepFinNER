from src.mda_extractor import extract_mda_from_text


def test_extract_mda():
    sample = "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations Lorem ipsum dolor sit amet. Item 8. Financial Statements."
    mda = extract_mda_from_text(sample)
    assert "Lorem ipsum" in mda
    assert "Item 8" not in mda 