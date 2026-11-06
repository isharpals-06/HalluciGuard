"""Unit tests for OKF schemas, generator, and validator."""
import pytest
from okf.schemas.okf_spec import OKFArtifact
from okf.generator.generator import create_okf_artifact
from okf.validator.validator import OKFValidator


def test_valid_okf_creation_and_validation():
    artifact = create_okf_artifact(
        artifact_id="OKF-TEST-001",
        statement="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        entity="Water",
        attribute="boiling_point",
        value=100,
        source_name="NIST Chemistry WebBook",
        reliability_estimate=0.99,
        verification_status="verified",
        verification_confidence=1.0,
    )
    assert artifact.identity.id == "OKF-TEST-001"
    assert artifact.content.value == 100

    validator = OKFValidator()
    is_valid, errors, parsed = validator.validate(artifact.model_dump())
    assert is_valid is True
    assert len(errors) == 0
    assert parsed.identity.id == "OKF-TEST-001"


def test_invalid_okf_rejected():
    validator = OKFValidator()
    # Missing required 'source' and 'provenance'
    invalid_dict = {
        "identity": {"id": "BAD-001", "okf_version": "1.0"},
        "content": {"statement": "Short", "entity": "Test", "attribute": "attr"},
    }
    is_valid, errors, parsed = validator.validate(invalid_dict)
    assert is_valid is False
    assert len(errors) > 0
    assert parsed is None


def test_temporal_consistency():
    validator = OKFValidator()
    artifact = create_okf_artifact(
        artifact_id="OKF-TEMP-001",
        statement="Test statement with bad temporal range.",
        entity="Test",
        attribute="test",
        valid_from="2025-01-01T00:00:00Z",
        valid_until="2020-01-01T00:00:00Z",  # Anomaly: until before from
    )
    is_valid, errors, parsed = validator.validate(artifact.model_dump())
    assert is_valid is False
    assert any("Temporal anomaly" in e for e in errors)
