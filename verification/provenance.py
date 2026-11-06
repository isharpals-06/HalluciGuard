"""
Provenance and Source Reliability Verification Service.
Evaluates source attribution, chain completeness, and institutional verification.
"""

from typing import Dict, Any
from okf.schemas.okf_spec import OKFArtifact


class ProvenanceAnalyzer:
    SOURCE_TYPE_WEIGHTS = {
        "primary": 1.0,
        "secondary": 0.80,
        "aggregated": 0.60,
        "synthesized": 0.40,
    }

    VERIFICATION_STATUS_WEIGHTS = {
        "verified": 1.0,
        "unverified": 0.50,
        "disputed": 0.20,
    }

    def evaluate(self, artifact: OKFArtifact) -> Dict[str, float]:
        """
        Computes source reliability and provenance confidence scores.
        """
        # 1. Base source reliability
        type_weight = self.SOURCE_TYPE_WEIGHTS.get(artifact.source.source_type.lower(), 0.5)
        raw_reliability = artifact.source.reliability_estimate
        source_reliability = 0.5 * type_weight + 0.5 * raw_reliability

        # 2. Provenance chain completeness
        chain_len = len(artifact.provenance.chain)
        chain_factor = min(1.0, 0.4 + 0.2 * chain_len)
        primary_factor = 1.0 if artifact.provenance.is_primary else 0.75

        # 3. Independent verification factor
        verif_factor = self.VERIFICATION_STATUS_WEIGHTS.get(
            artifact.verification.status.lower(), 0.5
        )
        verif_conf = artifact.verification.confidence

        provenance_confidence = (
            0.35 * chain_factor
            + 0.25 * primary_factor
            + 0.25 * verif_factor
            + 0.15 * verif_conf
        )

        return {
            "source_reliability": float(max(0.0, min(1.0, source_reliability))),
            "provenance_confidence": float(max(0.0, min(1.0, provenance_confidence))),
        }
