"""
Hybrid Retrieval Engine.
Fuses BM25 sparse lexical scores with Dense semantic vector scores
using calibrated linear combination and metadata-aware entity weighting.
"""

from typing import List, Tuple, Dict
from okf.schemas.okf_spec import OKFArtifact
from retrieval.sparse import BM25Retriever
from retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        artifacts: List[OKFArtifact],
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.6,
        rrf_k: int = 60,
        use_fallback_dense: bool = True,
        dense_mode: str = None,
    ):
        self.artifacts = artifacts
        self.alpha = alpha  # Weight for dense, (1 - alpha) for sparse
        self.rrf_k = rrf_k
        self.sparse_retriever = BM25Retriever(artifacts)
        self.dense_retriever = DenseRetriever(
            artifacts, model_name=dense_model_name, use_fallback=use_fallback_dense, mode=dense_mode
        )

    def retrieve(
        self, query: str, top_k: int = 5, entity_filter: str = None
    ) -> List[Tuple[OKFArtifact, float]]:
        """
        Retrieves top_k OKF artifacts using calibrated score fusion.
        Score represents actual relevance in [0, 1], not artificially scaled to 1.0.
        """
        if not self.artifacts:
            return []

        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)

        sparse_dict: Dict[str, Tuple[OKFArtifact, float]] = {
            art.identity.id: (art, score) for art, score in sparse_results
        }
        dense_dict: Dict[str, Tuple[OKFArtifact, float]] = {
            art.identity.id: (art, score) for art, score in dense_results
        }

        all_candidate_ids = set(sparse_dict.keys()).union(dense_dict.keys())
        if not all_candidate_ids:
            return []

        scored_candidates = []
        for art_id in all_candidate_ids:
            art = (
                dense_dict[art_id][0]
                if art_id in dense_dict
                else sparse_dict[art_id][0]
            )
            s_score = sparse_dict[art_id][1] if art_id in sparse_dict else 0.0
            d_score = dense_dict[art_id][1] if art_id in dense_dict else 0.0

            # Absolute linear fusion in [0, 1]
            fused = self.alpha * d_score + (1.0 - self.alpha) * s_score

            # Entity relevance boost if subject matches
            if entity_filter and entity_filter.lower() in art.content.entity.lower():
                fused = min(1.0, fused * 1.25)

            if fused > 0.10:
                scored_candidates.append((art, float(fused)))

        ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:top_k]
        return ranked
