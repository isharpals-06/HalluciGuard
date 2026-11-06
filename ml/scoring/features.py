"""
Multi-Signal Feature Extractor.
Extracts normalized 10-dimensional feature vectors from claims and verification outputs.
"""

from typing import Dict, Any, Optional
import numpy as np


class SignalFeatureExtractor:
    FEATURE_NAMES = [
        "semantic_similarity",
        "nli_entailment",
        "nli_contradiction",
        "nli_neutral",
        "evidence_coverage",
        "source_reliability",
        "provenance_confidence",
        "freshness_confidence",
        "retrieval_confidence",
        "claim_extraction_confidence",
        "evidence_disagreement",
        "evidence_count_normalized",
        "source_diversity",
        "mean_nli_entailment",
        "mean_nli_contradiction",
    ]

    def extract_vector(
        self,
        semantic_similarity: float,
        nli_probs: Dict[str, float],
        evidence_found: bool,
        retrieval_score: float,
        source_reliability: float,
        provenance_confidence: float,
        freshness_confidence: float,
        is_stale: bool,
        extraction_confidence: float = 0.95,
        evidence_disagreement: float = 0.0,
        evidence_count_normalized: float = 0.0,
        source_diversity: float = 0.0,
        mean_nli_entailment: Optional[float] = None,
        mean_nli_contradiction: Optional[float] = None,
    ) -> np.ndarray:
        """
        Produces a normalized 1D numpy array of shape (10,).
        """
        # If stale, discount the entailment signal and penalize freshness
        eff_freshness = freshness_confidence * (0.30 if is_stale else 1.0)
        eff_entailment = nli_probs.get("entailment", 0.0) * (0.20 if is_stale else 1.0)
        eff_contradiction = min(1.0, nli_probs.get("contradiction", 0.0) + (0.50 if is_stale else 0.0))

        feat = [
            float(np.clip(semantic_similarity, 0.0, 1.0)),
            float(np.clip(eff_entailment, 0.0, 1.0)),
            float(np.clip(eff_contradiction, 0.0, 1.0)),
            float(np.clip(nli_probs.get("neutral", 0.0), 0.0, 1.0)),
            1.0 if evidence_found else 0.0,
            float(np.clip(source_reliability, 0.0, 1.0)),
            float(np.clip(provenance_confidence, 0.0, 1.0)),
            float(np.clip(eff_freshness, 0.0, 1.0)),
            float(np.clip(retrieval_score, 0.0, 1.0)),
            float(np.clip(extraction_confidence, 0.0, 1.0)),
            float(np.clip(evidence_disagreement, 0.0, 1.0)),
            float(np.clip(evidence_count_normalized, 0.0, 1.0)),
            float(np.clip(source_diversity, 0.0, 1.0)),
            float(np.clip(
                nli_probs.get("entailment", 0.0) if mean_nli_entailment is None else mean_nli_entailment,
                0.0, 1.0,
            )),
            float(np.clip(
                nli_probs.get("contradiction", 0.0) if mean_nli_contradiction is None else mean_nli_contradiction,
                0.0, 1.0,
            )),
        ]
        return np.array(feat, dtype=np.float32)

    def to_feature_dict(self, vector: np.ndarray) -> Dict[str, float]:
        return {name: float(vector[i]) for i, name in enumerate(self.FEATURE_NAMES)}
