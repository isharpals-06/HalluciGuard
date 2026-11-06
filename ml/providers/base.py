"""
Base Abstract Provider for Claim Extraction and Verification.
Allows pluggable local and cloud model providers (Gemini, OpenAI, Claude, Local).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ProviderClaim(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: str = "factual"
    source_sentence: str
    entities: List[Dict[str, str]] = Field(default_factory=list)
    extraction_confidence: float = 0.90


class ProviderVerificationResult(BaseModel):
    status: str  # "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
    confidence: float
    explanation: str
    raw_response: Optional[Dict[str, Any]] = None


class VerificationProvider(ABC):
    """Abstract interface for modular verification providers."""

    def __init__(self, name: str, timeout_seconds: float = 10.0):
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if provider has required dependencies and credentials."""
        pass

    @abstractmethod
    def extract_claims(self, text: str) -> List[ProviderClaim]:
        """Extracts atomic factual claims from input text."""
        pass

    @abstractmethod
    def verify_claim(self, claim: str, evidence: str) -> ProviderVerificationResult:
        """Verifies an atomic claim against evidence statement."""
        pass
