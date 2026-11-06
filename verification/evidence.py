"""Evidence-level aggregation shared by runtime inference and training."""

from typing import Any, Dict, Iterable
import numpy as np

NLI_KEYS = ("entailment", "neutral", "contradiction")


def aggregate_evidence(items: Iterable[Dict[str, Any]], top_k: int,
                       relevance_threshold: float = 0.35,
                       nli_threshold: float = 0.60) -> Dict[str, Any]:
    """Aggregate already-scored evidence records without discarding item traces."""
    retained = [item for item in items if float(item.get("relevance", 0.0)) >= relevance_threshold]
    if not retained:
        return {
            "items": [],
            "nli_max": {"entailment": 0.02, "neutral": 0.96, "contradiction": 0.02},
            "mean_entailment": 0.0,
            "mean_contradiction": 0.0,
            "semantic_max": 0.02,
            "retrieval_max": 0.0,
            "evidence_condition": "NO_RELEVANT_EVIDENCE",
            "evidence_disagreement": 0.0,
            "evidence_count_normalized": 0.0,
            "source_diversity": 0.0,
        }

    nli_max = {key: max(float(item["nli"].get(key, 0.0)) for item in retained) for key in NLI_KEYS}
    support = any(float(item["nli"].get("entailment", 0.0)) >= nli_threshold for item in retained)
    contradiction = any(float(item["nli"].get("contradiction", 0.0)) >= nli_threshold for item in retained)
    condition = (
        "CONFLICTING_EVIDENCE" if support and contradiction
        else "SUPPORT_ONLY" if support
        else "CONTRADICTION_ONLY" if contradiction
        else "IRRELEVANT_EVIDENCE"
    )
    sources = {item.get("source_id") for item in retained if item.get("source_id")}
    return {
        "items": retained,
        "nli_max": nli_max,
        "mean_entailment": float(np.mean([item["nli"].get("entailment", 0.0) for item in retained])),
        "mean_contradiction": float(np.mean([item["nli"].get("contradiction", 0.0) for item in retained])),
        "semantic_max": max(float(item.get("semantic", 0.0)) for item in retained),
        "retrieval_max": max(float(item.get("relevance", 0.0)) for item in retained),
        "evidence_condition": condition,
        "evidence_disagreement": float(support and contradiction),
        "evidence_count_normalized": min(1.0, len(retained) / max(int(top_k), 1)),
        "source_diversity": len(sources) / len(retained),
    }
