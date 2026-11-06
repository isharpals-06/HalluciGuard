"""Unit tests for retrieval and verification services."""
import pytest
from okf.generator.generator import create_okf_artifact
from retrieval.sparse import BM25Retriever
from ml.nli.nli_engine import NLIEngine
from verification.provenance import ProvenanceAnalyzer
from verification.freshness import FreshnessAnalyzer


@pytest.fixture
def sample_artifacts():
    art1 = create_okf_artifact(
        artifact_id="OKF-1",
        statement="Apollo 11 landed on the moon on July 20, 1969.",
        entity="Apollo 11",
        attribute="landing_date",
        value="1969-07-20",
        source_name="NASA",
        source_type="primary",
        reliability_estimate=1.0,
        verification_status="verified",
    )
    art2 = create_okf_artifact(
        artifact_id="OKF-2",
        statement="Rishi Sunak served as Prime Minister until July 5, 2024.",
        entity="Rishi Sunak",
        attribute="term",
        valid_until="2024-07-05T00:00:00Z",
        freshness_state="STALE",
    )
    return [art1, art2]


def test_bm25_retrieval(sample_artifacts):
    bm25 = BM25Retriever(sample_artifacts)
    results = bm25.retrieve("Apollo 11 lunar landing", top_k=1)
    assert len(results) == 1
    assert results[0][0].identity.id == "OKF-1"
    assert results[0][1] > 0.0


def test_nli_date_contradiction():
    nli = NLIEngine()
    premise = "Apollo 11 landed on the moon on July 20, 1969."
    hypothesis = "The Apollo 11 moon landing occurred in 1974."
    probs = nli.predict(premise, hypothesis)

    assert probs["contradiction"] > 0.80
    assert probs["entailment"] < 0.10


def test_provenance_analyzer(sample_artifacts):
    analyzer = ProvenanceAnalyzer()
    res = analyzer.evaluate(sample_artifacts[0])
    assert res["source_reliability"] > 0.90
    assert res["provenance_confidence"] > 0.80


def test_freshness_analyzer(sample_artifacts):
    analyzer = FreshnessAnalyzer()
    claim_stale = "Rishi Sunak is currently the Prime Minister of the UK."
    res = analyzer.evaluate(claim_stale, sample_artifacts[1])
    assert res["freshness_state"] == "STALE"
    assert res["is_stale"] is True
