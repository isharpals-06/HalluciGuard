"""
Semantic Similarity Engine.
Computes fine-grained semantic overlap between claims and evidence text.
"""

from typing import Optional
import numpy as np


class SemanticSimilarityEngine:
    def __init__(self, dense_retriever=None):
        self.dense_retriever = dense_retriever

    def compute_similarity(self, claim_text: str, evidence_text: str) -> float:
        """
        Computes cosine similarity between claim and evidence statements.
        """
        if not claim_text or not evidence_text:
            return 0.0

        if self.dense_retriever and self.dense_retriever.model and not self.dense_retriever.use_fallback:
            try:
                embs = self.dense_retriever.model.encode(
                    [claim_text, evidence_text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                sim = float(np.dot(embs[0], embs[1]))
                return max(0.0, min(1.0, sim))
            except Exception:
                pass

        # Robust word n-gram Jaccard / token overlap fallback
        import re
        tokens_claim = set(re.findall(r"\w+", claim_text.lower()))
        tokens_ev = set(re.findall(r"\w+", evidence_text.lower()))

        if not tokens_claim or not tokens_ev:
            return 0.0

        intersection = len(tokens_claim.intersection(tokens_ev))
        union = len(tokens_claim.union(tokens_ev))
        overlap = intersection / union if union > 0 else 0.0
        return float(min(1.0, overlap * 1.5))
