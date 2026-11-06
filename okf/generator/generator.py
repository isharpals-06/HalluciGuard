"""
OKF Knowledge Artifact Generator.
Facilitates the creation of validated OKF objects from structured parameters.
"""

from datetime import datetime, timezone
from typing import Optional, List, Any
from okf.schemas.okf_spec import (
    OKFArtifact,
    OKFIdentity,
    OKFContent,
    OKFSource,
    OKFProvenance,
    OKFTemporal,
    OKFVerification,
    OKFLifecycle,
)


def create_okf_artifact(
    artifact_id: str,
    statement: str,
    entity: str,
    attribute: str,
    value: Any = None,
    category: str = "general",
    source_name: str = "Primary Archive",
    source_url: Optional[str] = None,
    source_type: str = "primary",
    reliability_estimate: float = 0.95,
    provenance_chain: Optional[List[str]] = None,
    valid_from: Optional[str] = None,
    valid_until: Optional[str] = None,
    temporal_scope: str = "permanent",
    verification_status: str = "verified",
    verification_confidence: float = 0.98,
    freshness_state: str = "CURRENT",
) -> OKFArtifact:
    """Helper factory for generating standard OKF artifacts."""
    now_iso = datetime.now(timezone.utc).isoformat()

    return OKFArtifact(
        identity=OKFIdentity(id=artifact_id, okf_version="1.0"),
        content=OKFContent(
            statement=statement,
            entity=entity,
            attribute=attribute,
            value=value,
            category=category,
        ),
        source=OKFSource(
            name=source_name,
            url=source_url,
            source_type=source_type,
            reliability_estimate=reliability_estimate,
        ),
        provenance=OKFProvenance(
            chain=provenance_chain or [source_name],
            is_primary=(source_type == "primary"),
            recorded_at=now_iso,
        ),
        temporal=OKFTemporal(
            valid_from=valid_from,
            valid_until=valid_until,
            temporal_scope=temporal_scope,
        ),
        verification=OKFVerification(
            status=verification_status,
            confidence=verification_confidence,
            verifier="official_curator",
        ),
        lifecycle=OKFLifecycle(
            freshness_state=freshness_state,
            ingested_at=now_iso,
        ),
    )
