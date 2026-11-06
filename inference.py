"""
Production & Research Inference Pipeline for HalluciGuard-OKF.
Implements Uncertainty-Aware Atomic Claim Verification with Selective Abstention.
Provides both programmatic Python interface and command-line execution.

Usage:
    python inference.py --query "Who founded Tesla?" --response "Tesla was founded in 2008 by Elon Musk."
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import yaml

from okf.schemas.okf_spec import OKFArtifact
from knowledge.store import OKFKnowledgeStore
from ml.claim_extraction.extractor import ClaimExtractor, ExtractedClaim
from retrieval.hybrid import HybridRetriever
from ml.nli.nli_engine import NLIEngine
from verification.semantic import SemanticSimilarityEngine
from verification.provenance import ProvenanceAnalyzer
from verification.freshness import FreshnessAnalyzer
from verification.evidence import aggregate_evidence
from ml.scoring.features import SignalFeatureExtractor
from ml.scoring.models import MultiSignalScorer, ProductionScorer, SafeFeatureScaler
from ml.providers import get_provider


class HalluciGuardPipeline:
    """
    End-to-end inference orchestrator for HalluciGuard-OKF.
    Executes:
    1. Input validation & ReDoS mitigation
    2. Atomic claim extraction
    3. OKF evidence retrieval (hybrid sparse + dense with entity routing)
    4. Multi-signal verification (NLI, semantic, provenance, freshness)
    5. Scaled feature vector generation (safe serialization)
    6. Uncertainty estimation & selective prediction (abstention logic)
    7. Evidence-grounded explanation synthesis
    8. Response-level metric aggregation
    """

    def __init__(
        self,
        knowledge_path: str = "knowledge/artifacts/curated_knowledge.json",
        weights_path: str = "models/multisignal_head/best_model.pt",
        scaler_path: str = "models/multisignal_head/feature_scaler.json",
        min_relevance_threshold: float = 0.35,
        abstain_threshold: Optional[float] = None,
        provider_name: str = "local",
        config_path: str = "configs/model_config.yaml",
    ):
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        retrieval_config = config.get("retrieval", {})
        embedding_config = config.get("embedding", {})
        nli_config = config.get("nli", {})
        scoring_config = config.get("scoring", {})
        self.config = config
        self.min_relevance_threshold = float(retrieval_config.get("min_relevance_threshold", min_relevance_threshold))
        configured_abstain_threshold = float(scoring_config.get("abstain_threshold", 0.0))
        self.abstain_threshold = (
            float(abstain_threshold) if abstain_threshold is not None else configured_abstain_threshold
        )
        self.knowledge_store = OKFKnowledgeStore(knowledge_path)
        all_artifacts = self.knowledge_store.get_all()

        self.retriever = HybridRetriever(
            all_artifacts,
            dense_model_name=embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            alpha=float(retrieval_config.get("hybrid_alpha", 0.6)),
            dense_mode=embedding_config.get("mode", "tfidf"),
        )
        self.claim_extractor = ClaimExtractor()
        self.nli_engine = NLIEngine(
            model_name=nli_config.get("model_name"),
            device=nli_config.get("device", "cpu"),
            temperature=float(nli_config.get("temperature", 1.0)),
            mode=nli_config.get("mode", "rule_based"),
        )
        self.semantic_engine = SemanticSimilarityEngine(self.retriever.dense_retriever)
        self.provenance_analyzer = ProvenanceAnalyzer()
        self.freshness_analyzer = FreshnessAnalyzer()
        self.feature_extractor = SignalFeatureExtractor()
        self.provider = get_provider(provider_name)

        # Load trained neural head with safe weights
        self.scorer = ProductionScorer(
            model=MultiSignalScorer(
                input_dim=int(scoring_config.get("num_features", 10)),
                hidden_dims=tuple(scoring_config.get("hidden_dims", [64, 32])),
                dropout=float(scoring_config.get("dropout", 0.2)),
                num_classes=int(scoring_config.get("num_classes", 3)),
            ),
            weights_path=weights_path if Path(weights_path).exists() or Path(weights_path).with_suffix(".safetensors").exists() else None,
            abstain_threshold=self.abstain_threshold,
        )
        metadata_path = Path(weights_path).with_name("model_metadata.json")
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    model_metadata = json.load(metadata_file)
                repro = model_metadata.get("reproducibility", {})
                self.scorer.temperature_scale = float(repro.get("posthoc_temperature", 1.0))
                selected_threshold = repro.get("selected_abstention_threshold")
                if abstain_threshold is None and selected_threshold is not None:
                    self.abstain_threshold = float(selected_threshold)
                    self.scorer.abstain_threshold = self.abstain_threshold
            except (OSError, ValueError, TypeError):
                self.scorer.temperature_scale = 1.0

        # Load scaler safely from JSON first
        self.scaler = None
        json_scaler = Path(scaler_path)
        if json_scaler.exists() and json_scaler.suffix == ".json":
            self.scaler = SafeFeatureScaler(str(json_scaler))
        elif Path(scaler_path).exists():
            raise ValueError("Only JSON feature scalers are supported; pickle scaler loading is disabled.")

    def analyze(
        self,
        response_text: str,
        query: Optional[str] = None,
        domain: str = "general",
        language: str = "en",
        abstain_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete verification analysis for a given query and response.
        """
        start_time = datetime.now(timezone.utc)
        request_id = f"REQ-{uuid.uuid4().hex[:12].upper()}"
        analysis_id = str(uuid.uuid4())

        if abstain_threshold is not None:
            active_abstain_thresh = abstain_threshold
        else:
            active_abstain_thresh = self.abstain_threshold

        self.scorer.abstain_threshold = active_abstain_thresh

        if not response_text or not response_text.strip():
            return {
                "analysis_id": analysis_id,
                "request_id": request_id,
                "error": "Empty response text provided",
                "overall_status": "ERROR",
            }

        # Step 1: Atomic Claim Extraction
        extracted_claims: List[ExtractedClaim] = self.claim_extractor.extract_claims(
            response_text, query=query
        )

        analyzed_claims = []
        contradicted_count = 0
        supported_count = 0
        unverified_count = 0
        abstained_count = 0
        evidence_found_count = 0
        confidences = []
        uncertainties = []

        # Step 2: Per-Claim Verification
        for claim in extracted_claims:
            # Pick primary PROPER_NOUN entity for targeted retrieval
            entity_filter_val = None
            for ent in claim.entities:
                if ent.get("type") == "PROPER_NOUN":
                    entity_filter_val = ent.get("name")
                    break

            # 2a. Evidence Retrieval
            retrieved = self.retriever.retrieve(
                claim.claim_text,
                top_k=int(self.config.get("retrieval", {}).get("top_k", 5)),
                entity_filter=entity_filter_val,
            )

            relevant_items = [
                (artifact, score) for artifact, score in retrieved
                if score >= self.min_relevance_threshold
            ]
            has_evidence = bool(relevant_items)
            top_artifact, top_ret_score = relevant_items[0] if has_evidence else (None, 0.0)
            evidence_outputs = []

            if has_evidence:
                evidence_found_count += 1
                for rank, (artifact, relevance) in enumerate(relevant_items, start=1):
                    text = artifact.content.statement
                    item_nli = self.nli_engine.predict(text, claim.claim_text)
                    item_semantic = self.semantic_engine.compute_similarity(claim.claim_text, text)
                    item_provenance = self.provenance_analyzer.evaluate(artifact)
                    item_freshness = self.freshness_analyzer.evaluate(claim.claim_text, artifact)
                    evidence_outputs.append({
                        "artifact": artifact,
                        "text": text,
                        "relevance": float(relevance),
                        "nli": item_nli,
                        "semantic": float(item_semantic),
                        "provenance": item_provenance,
                        "freshness": item_freshness,
                        "rank": rank,
                    })

                evidence_text = top_artifact.content.statement
                aggregated = aggregate_evidence(
                    [{**item, "source_id": item["artifact"].source.name} for item in evidence_outputs],
                    top_k=int(self.config.get("retrieval", {}).get("top_k", 5)),
                    relevance_threshold=self.min_relevance_threshold,
                )
                evidence_outputs = aggregated["items"]
                sim_score = aggregated["semantic_max"]
                nli_probs = aggregated["nli_max"]
                top_ret_score = aggregated["retrieval_max"]
                source_rel = max(item["provenance"]["source_reliability"] for item in evidence_outputs)
                prov_conf = max(item["provenance"]["provenance_confidence"] for item in evidence_outputs)
                top_provenance = evidence_outputs[0]["provenance"]
                top_freshness = evidence_outputs[0]["freshness"]
                freshness_state = top_freshness["freshness_state"]
                freshness_conf = min(item["freshness"]["freshness_confidence"] for item in evidence_outputs)
                is_stale = any(item["freshness"]["is_stale"] for item in evidence_outputs)
                evidence_condition = aggregated["evidence_condition"]
                evidence_disagreement = aggregated["evidence_disagreement"]
                evidence_count_normalized = aggregated["evidence_count_normalized"]
                source_diversity = aggregated["source_diversity"]
                mean_entailment = aggregated["mean_entailment"]
                mean_contradiction = aggregated["mean_contradiction"]
                evidence_payload = [{
                    "evidence_id": f"EV-{item['artifact'].identity.id}",
                    "okf_object_id": item["artifact"].identity.id,
                    "source_id": item["artifact"].source.name,
                    "text": item["text"],
                    "relevance_score": round(item["relevance"], 4),
                    "rank": item["rank"],
                    "nli": {key: round(float(value), 4) for key, value in item["nli"].items()},
                } for item in evidence_outputs]
                provenance_payload = {
                    "source_name": top_artifact.source.name,
                    "source_type": top_artifact.source.source_type,
                    "verified": (top_artifact.verification.status == "verified"),
                    "ingested_at": top_artifact.lifecycle.ingested_at,
                    "freshness_state": freshness_state,
                }
            else:
                # No evidence retrieved above relevance threshold
                evidence_text = ""
                top_ret_score = 0.0
                sim_score = 0.02
                nli_probs = {"entailment": 0.02, "neutral": 0.96, "contradiction": 0.02}
                source_rel = 0.50
                prov_conf = 0.20
                freshness_state = "UNKNOWN"
                freshness_conf = 0.50
                is_stale = False
                evidence_payload = []
                provenance_payload = {}
                evidence_condition = "NO_RELEVANT_EVIDENCE" if not retrieved else "IRRELEVANT_EVIDENCE"
                evidence_disagreement = 0.0
                evidence_count_normalized = 0.0
                source_diversity = 0.0
                mean_entailment = 0.0
                mean_contradiction = 0.0

            # Step 3: Feature Vector Generation & Normalization
            feat_vec = self.feature_extractor.extract_vector(
                semantic_similarity=sim_score,
                nli_probs=nli_probs,
                evidence_found=has_evidence,
                retrieval_score=top_ret_score,
                source_reliability=source_rel,
                provenance_confidence=prov_conf,
                freshness_confidence=freshness_conf,
                is_stale=is_stale,
                extraction_confidence=claim.extraction_confidence,
                evidence_disagreement=evidence_disagreement,
                evidence_count_normalized=evidence_count_normalized,
                source_diversity=source_diversity,
                mean_nli_entailment=mean_entailment,
                mean_nli_contradiction=mean_contradiction,
            )

            if self.scaler:
                norm_feat_vec = self.scaler.transform(feat_vec.reshape(1, -1))[0]
            else:
                norm_feat_vec = feat_vec

            # Step 4: Model Inference with Epistemic Uncertainty & Selective Abstention
            decision = self.scorer.predict_single(
                norm_feat_vec,
                evidence_found=has_evidence,
                retrieval_score=top_ret_score,
            )

            status = decision["status"]
            conf = decision["confidence"]
            flags = decision["flags"]
            hallucination_score = decision["hallucination_score"]
            uncertainty_score = decision["uncertainty_score"]
            abstained = decision["abstained"]
            abstention_reason = decision["abstention_reason"]

            if evidence_condition == "CONFLICTING_EVIDENCE":
                status = "UNVERIFIED"
                abstained = True
                abstention_reason = "Retrieved evidence contains material support and contradiction; human review is required."
                flags = list(flags) + ["CONFLICTING_EVIDENCE", "HUMAN_REVIEW"]

            # Grounding alignment for high-confidence NLI signals with verified evidence
            if evidence_condition != "CONFLICTING_EVIDENCE" and has_evidence and nli_probs.get("contradiction", 0) > 0.80:
                status = "CONTRADICTED"
                conf = max(conf, float(nli_probs["contradiction"]))
                hallucination_score = max(hallucination_score, 0.85)
                abstained = False
            elif evidence_condition != "CONFLICTING_EVIDENCE" and has_evidence and nli_probs.get("entailment", 0) > 0.80 and sim_score >= 0.40:
                status = "SUPPORTED"
                conf = max(conf, float(nli_probs["entailment"]))
                hallucination_score = min(hallucination_score, 0.15)
                abstained = False

            confidences.append(conf)
            uncertainties.append(uncertainty_score)

            if abstained:
                abstained_count += 1

            if status == "CONTRADICTED":
                contradicted_count += 1
            elif status == "SUPPORTED":
                supported_count += 1
            else:
                status = "UNVERIFIED"
                unverified_count += 1

            # Step 5: Structured Evidence-Grounded Explanation
            explanation = self._build_explanation(
                claim_text=claim.claim_text,
                status=status,
                has_evidence=has_evidence,
                evidence_text=evidence_text,
                nli_probs=nli_probs,
                provenance=provenance_payload,
                is_stale=is_stale,
                abstained=abstained,
                abstention_reason=abstention_reason,
            )

            analyzed_claims.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "status": status,
                "claim_status": "INSUFFICIENT_EVIDENCE" if status == "UNVERIFIED" else status,
                "evidence_condition": evidence_condition,
                "confidence": conf,
                "uncertainty_score": uncertainty_score,
                "abstained": abstained,
                "abstention_reason": abstention_reason,
                "flags": flags,
                "hallucination_score": hallucination_score,
                "evidence": evidence_payload,
                "provenance": provenance_payload,
                "signals": {
                    "semantic_similarity": round(float(sim_score), 4),
                    "nli_entailment": round(float(nli_probs.get("entailment", 0)), 4),
                    "nli_contradiction": round(float(nli_probs.get("contradiction", 0)), 4),
                    "source_reliability": round(float(source_rel), 4),
                    "provenance_confidence": round(float(prov_conf), 4),
                    "freshness_confidence": round(float(freshness_conf), 4),
                    "retrieval_confidence": round(float(top_ret_score), 4),
                    "claim_extraction_confidence": round(float(claim.extraction_confidence), 4),
                    "evidence_disagreement": int(evidence_condition == "CONFLICTING_EVIDENCE"),
                },
                "explanation": explanation,
            })

        # Step 6: Response-Level Metric Aggregation
        total_claims = len(analyzed_claims)
        hallucination_rate = contradicted_count / total_claims if total_claims > 0 else 0.0
        evidence_coverage = evidence_found_count / total_claims if total_claims > 0 else 0.0
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.0
        coverage_rate = (total_claims - abstained_count) / total_claims if total_claims > 0 else 1.0

        if contradicted_count > 0:
            overall_status = "HALLUCINATED"
            composite_hallucination_score = min(1.0, 0.40 + 0.60 * hallucination_rate)
        elif unverified_count == total_claims:
            overall_status = "UNVERIFIED"
            composite_hallucination_score = 0.50
        elif supported_count == total_claims:
            overall_status = "VERIFIED"
            composite_hallucination_score = 0.05
        else:
            overall_status = "PARTIALLY_VERIFIED"
            composite_hallucination_score = round(0.50 * (unverified_count / total_claims), 4)

        elapsed_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        return {
            "analysis_id": analysis_id,
            "request_id": request_id,
            "overall_status": overall_status,
            "hallucination_score": round(float(composite_hallucination_score), 4),
            "language": language,
            "domain": domain,
            "processing_time_ms": elapsed_ms,
            "claims": analyzed_claims,
            "metrics": {
                "total_claims": total_claims,
                "supported_claims": supported_count,
                "contradicted_claims": contradicted_count,
                "unverified_claims": unverified_count,
                "abstained_claims": abstained_count,
                "coverage_rate": round(coverage_rate, 4),
                "evidence_coverage": round(evidence_coverage, 4),
                "hallucination_rate": round(hallucination_rate, 4),
                "average_confidence": round(avg_confidence, 4),
                "mean_uncertainty": round(mean_uncertainty, 4),
            },
            "okf_version": "1.0",
            "execution_modes": {
                "nli": self.nli_engine.mode,
                "dense_retrieval": self.retriever.dense_retriever.mode,
            },
        }

    def _build_explanation(
        self,
        claim_text: str,
        status: str,
        has_evidence: bool,
        evidence_text: str,
        nli_probs: Dict[str, float],
        provenance: Dict[str, Any],
        is_stale: bool,
        abstained: bool = False,
        abstention_reason: Optional[str] = None,
    ) -> str:
        """Constructs an auditable, evidence-grounded explanation."""
        if not has_evidence:
            return (
                f"No verified knowledge objects were found in the OKF repository for the claim '{claim_text}'. "
                "Per system policy, absence of evidence results in UNVERIFIED rather than falsification."
            )

        src = provenance.get("source_name", "Curated Repository")
        p_con = nli_probs.get("contradiction", 0.0)
        p_ent = nli_probs.get("entailment", 0.0)

        if is_stale:
            return (
                f"The claim '{claim_text}' asserts a current status that is contradicted by temporal records. "
                f"Verified historical evidence from {src} indicates this state has expired ({evidence_text})."
            )

        if abstained:
            return (
                f"The system abstained from making a decisive judgment on '{claim_text}' because "
                f"{abstention_reason} Retrieved evidence from {src}: '{evidence_text}'."
            )

        if status == "CONTRADICTED":
            return (
                f"The claim '{claim_text}' is directly contradicted by verified evidence from {src}: "
                f"'{evidence_text}'. NLI contradiction probability is {p_con:.2f}."
            )
        elif status == "SUPPORTED":
            return (
                f"The claim '{claim_text}' is confirmed by verified evidence from {src}: "
                f"'{evidence_text}'. NLI entailment probability is {p_ent:.2f}."
            )
        else:
            return (
                f"Evidence was retrieved ('{evidence_text}'), but semantic entailment was ambiguous "
                f"(Entailment: {p_ent:.2f}, Contradiction: {p_con:.2f}). Classified as UNVERIFIED."
            )


def main():
    parser = argparse.ArgumentParser(description="Run HalluciGuard-OKF Inference Pipeline")
    parser.add_argument("--query", type=str, default="Who founded Tesla?", help="User query")
    parser.add_argument(
        "--response",
        type=str,
        default="Tesla was founded in 2008 by Elon Musk.",
        help="LLM response text",
    )
    parser.add_argument("--input", type=str, default=None, help="Alias for --response")
    parser.add_argument("--domain", type=str, default="business", help="Domain category")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--abstain-threshold", type=float, default=None, help="Override validation-selected abstention threshold [0.0 - 1.0]")
    parser.add_argument("--provider", type=str, default="local", help="Provider: local, gemini, openai")
    parser.add_argument("--forced", action="store_true", help="Force classification without abstention")

    args = parser.parse_args()
    resp_text = args.input if args.input else args.response
    abstain_thresh = 0.0 if args.forced else args.abstain_threshold

    pipeline = HalluciGuardPipeline(
        abstain_threshold=abstain_thresh,
        provider_name=args.provider,
    )
    result = pipeline.analyze(
        response_text=resp_text,
        query=args.query,
        domain=args.domain,
        abstain_threshold=abstain_thresh,
    )

    formatted_json = json.dumps(result, indent=2)
    print(formatted_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted_json)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
