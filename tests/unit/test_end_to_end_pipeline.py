"""End-to-end integration and pipeline tests."""
import pytest
from inference import HalluciGuardPipeline


@pytest.fixture
def pipeline():
    return HalluciGuardPipeline()


def test_supported_claim_pipeline(pipeline):
    result = pipeline.analyze(
        response_text="The speed of light in a vacuum is exactly 299,792,458 meters per second.",
        query="What is the speed of light in vacuum?",
    )
    assert result["overall_status"] == "VERIFIED"
    assert result["metrics"]["supported_claims"] >= 1
    assert result["metrics"]["contradicted_claims"] == 0
    assert result["hallucination_score"] <= 0.20


def test_contradicted_claim_pipeline(pipeline):
    result = pipeline.analyze(
        response_text="The capital city of Australia is Sydney.",
        query="What is the capital of Australia?",
    )
    assert result["overall_status"] == "HALLUCINATED"
    assert result["metrics"]["contradicted_claims"] >= 1
    assert result["hallucination_score"] >= 0.50
    assert "Canberra" in result["claims"][0]["explanation"]


def test_unverified_claim_pipeline(pipeline):
    result = pipeline.analyze(
        response_text="Dr. Arthur Pendelton invented the acoustic resonator at ETH Zurich in 1994.",
        query="Tell me about Arthur Pendelton",
    )
    assert result["overall_status"] == "UNVERIFIED"
    assert result["metrics"]["unverified_claims"] >= 1
    assert result["claims"][0]["status"] == "UNVERIFIED"
