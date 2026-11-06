"""
Local Offline Provider.
Uses local linguistic rules, semantic models, and NLI verification.
Always available without external credentials.
"""

from typing import List
from ml.providers.base import VerificationProvider, ProviderClaim, ProviderVerificationResult
from ml.claim_extraction.extractor import ClaimExtractor
from ml.nli.nli_engine import NLIEngine


class LocalProvider(VerificationProvider):
    """Local, offline, zero-network verification provider."""

    def __init__(self, extractor: ClaimExtractor = None, nli_engine: NLIEngine = None):
        super().__init__(name="local", timeout_seconds=0.0)
        self.extractor = extractor or ClaimExtractor()
        self.nli_engine = nli_engine or NLIEngine()

    def is_available(self) -> bool:
        return True

    def extract_claims(self, text: str) -> List[ProviderClaim]:
        extracted = self.extractor.extract_claims(text)
        return [
            ProviderClaim(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                claim_type=c.claim_type,
                source_sentence=c.source_sentence,
                entities=c.entities,
                extraction_confidence=c.extraction_confidence,
            )
            for c in extracted
        ]

    def verify_claim(self, claim: str, evidence: str) -> ProviderVerificationResult:
        if not evidence or not evidence.strip():
            return ProviderVerificationResult(
                status="INSUFFICIENT_EVIDENCE",
                confidence=0.90,
                explanation="No reference evidence available to verify claim.",
            )

        nli_probs = self.nli_engine.predict(evidence, claim)
        p_ent = nli_probs.get("entailment", 0.0)
        p_con = nli_probs.get("contradiction", 0.0)
        p_neu = nli_probs.get("neutral", 0.0)

        if p_con > 0.65:
            return ProviderVerificationResult(
                status="CONTRADICTED",
                confidence=float(p_con),
                explanation=f"Contradicted by evidence with confidence {p_con:.2f}",
                raw_response=nli_probs,
            )
        elif p_ent > 0.60:
            return ProviderVerificationResult(
                status="SUPPORTED",
                confidence=float(p_ent),
                explanation=f"Supported by evidence with confidence {p_ent:.2f}",
                raw_response=nli_probs,
            )
        else:
            return ProviderVerificationResult(
                status="INSUFFICIENT_EVIDENCE",
                confidence=float(p_neu),
                explanation=f"Evidence is ambiguous or inconclusive (Neutral: {p_neu:.2f})",
                raw_response=nli_probs,
            )
