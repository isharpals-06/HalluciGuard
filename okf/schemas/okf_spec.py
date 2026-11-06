"""
Open Knowledge Format (OKF) v1.0 Standard Specifications and Pydantic Schemas.
Enforces structured, provenance-aware representation of factual knowledge.
"""

from typing import List, Optional, Any, Literal
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime


class OKFIdentity(BaseModel):
    id: str = Field(..., description="Unique knowledge object identifier, e.g. OKF-SCI-001")
    okf_version: Literal["1.0", "1.0.0"] = Field("1.0", description="OKF specification version")


class OKFContent(BaseModel):
    statement: str = Field(..., min_length=5, description="Factual assertion statement")
    entity: str = Field(..., description="Primary subject entity")
    attribute: str = Field(..., description="Attribute or relation of entity")
    value: Optional[Any] = Field(None, description="Normalized attribute value")
    category: Optional[str] = Field("general", description="Domain or topic category")


class OKFSource(BaseModel):
    name: str = Field(..., description="Name of originating publisher or archive")
    url: Optional[str] = Field(None, description="Canonical URL reference")
    source_type: Literal["primary", "secondary", "aggregated", "synthesized"] = Field(
        ..., description="Category of the source"
    )
    reliability_estimate: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated reliability prior score"
    )
    license: Optional[str] = Field("CC-BY-4.0", description="Data license")


class OKFProvenance(BaseModel):
    chain: List[str] = Field(default_factory=list, description="Chain of attribution citations")
    is_primary: bool = Field(True, description="Whether the evidence traces directly to primary source")
    recorded_at: str = Field(..., description="ISO 8601 recording timestamp")


class OKFTemporal(BaseModel):
    valid_from: Optional[str] = Field(None, description="ISO timestamp from which the fact became valid")
    valid_until: Optional[str] = Field(None, description="ISO timestamp until which fact is valid (null if current)")
    temporal_scope: Literal["current", "historical", "permanent", "transient"] = Field(
        "permanent", description="Temporal lifecycle scope"
    )


class OKFVerification(BaseModel):
    status: Literal["verified", "unverified", "disputed"] = Field(
        ..., description="Independent verification status"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in correctness")
    verifier: Optional[str] = Field("expert_consensus", description="Entity that verified the fact")


class OKFLifecycle(BaseModel):
    freshness_state: Literal["CURRENT", "HISTORICAL", "STALE", "UNKNOWN"] = Field(
        "CURRENT", description="Knowledge freshness state"
    )
    ingested_at: str = Field(..., description="ISO timestamp of system ingestion")


class OKFArtifact(BaseModel):
    """
    Complete Open Knowledge Format (OKF) Knowledge Object.
    """
    identity: OKFIdentity
    content: OKFContent
    source: OKFSource
    provenance: OKFProvenance
    temporal: OKFTemporal
    verification: OKFVerification
    lifecycle: OKFLifecycle

    def to_summary_text(self) -> str:
        """Generates textual representation for retrieval and NLI premise evaluation."""
        return (
            f"{self.content.statement} (Entity: {self.content.entity}, "
            f"Source: {self.source.name}, Status: {self.verification.status}, "
            f"Freshness: {self.lifecycle.freshness_state})"
        )
