"""Unit tests for atomic claim extraction."""
import pytest
from ml.claim_extraction.extractor import ClaimExtractor


def test_atomic_decomposition():
    extractor = ClaimExtractor()
    text = "Tesla was founded in 2008 by Elon Musk."
    claims = extractor.extract_claims(text)

    assert len(claims) == 2
    assert "in 2008" in claims[0].claim_text
    assert "by Elon Musk" in claims[1].claim_text
    assert claims[0].claim_type == "temporal"
    assert claims[1].claim_type == "attribution"


def test_multi_sentence_extraction():
    extractor = ClaimExtractor()
    text = "Gold has an atomic number of 79. It is a transition metal."
    claims = extractor.extract_claims(text)

    assert len(claims) == 2
    assert claims[0].claim_type == "numerical"


def test_empty_input():
    extractor = ClaimExtractor()
    assert extractor.extract_claims("") == []
    assert extractor.extract_claims("   ") == []
