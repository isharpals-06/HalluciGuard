"""
Dataset Ingestion, Preprocessing, and Split Pipeline.
Guarantees zero data leakage: feature scalers are fitted strictly on the training partition.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from ml.scoring.features import SignalFeatureExtractor
from ml.nli.nli_engine import NLIEngine
from verification.semantic import SemanticSimilarityEngine
from verification.provenance import ProvenanceAnalyzer
from verification.freshness import FreshnessAnalyzer
from okf.schemas.okf_spec import OKFArtifact
from knowledge.store import OKFKnowledgeStore
from retrieval.hybrid import HybridRetriever


LABEL_MAP = {"SUPPORTED": 0, "CONTRADICTED": 1, "UNVERIFIED": 2}
INV_LABEL_MAP = {0: "SUPPORTED", 1: "CONTRADICTED", 2: "UNVERIFIED"}


class HallucinationDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, risk_scores: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.risk_scores = torch.from_numpy(risk_scores.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.risk_scores[idx]


class BenchmarkDataLoader:
    def __init__(
        self,
        benchmark_path: str = "datasets/benchmark/benchmark_data.json",
        knowledge_store = None,
        random_seed: int = 42,
        evidence_mode: str = "retrieval",
        knowledge_path: str = "knowledge/artifacts/curated_knowledge.json",
        nli_mode: str = "rule_based",
        nli_model_name: str = "roberta-large-mnli",
        dense_mode: str = "tfidf",
        top_k: int = 5,
        min_relevance_threshold: float = 0.35,
    ):
        self.benchmark_path = Path(benchmark_path)
        self.knowledge_store = knowledge_store
        self.random_seed = random_seed
        if evidence_mode not in {"retrieval", "oracle_evidence"}:
            raise ValueError("evidence_mode must be 'retrieval' or 'oracle_evidence'.")
        self.evidence_mode = evidence_mode
        self.feature_extractor = SignalFeatureExtractor()
        self.nli_engine = NLIEngine(model_name=nli_model_name, mode=nli_mode)
        self.knowledge_store = knowledge_store or OKFKnowledgeStore(knowledge_path)
        self.retriever = HybridRetriever(
            self.knowledge_store.get_all(), dense_mode=dense_mode
        )
        self.top_k = int(top_k)
        self.min_relevance_threshold = float(min_relevance_threshold)
        self.semantic_engine = SemanticSimilarityEngine(self.retriever.dense_retriever)
        self.scaler = StandardScaler()

    def load_and_preprocess(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Loads benchmark JSON, retrieves evidence from the configured knowledge store,
        splits by source scenario, and fits normalization on training rows only.
        """
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            raw_examples = json.load(f)

        features_list = []
        labels_list = []
        risk_list = []
        metadata_list = []
        group_list = []

        seen_claims = set()

        for ex in raw_examples:
            for c in ex.get("claims", []):
                claim_text = c["claim_text"].strip()
                # Deduplication check
                if claim_text in seen_claims:
                    continue
                seen_claims.add(claim_text)

                gold_label_str = c["gold_label"].upper()
                if gold_label_str not in LABEL_MAP:
                    continue
                label_idx = LABEL_MAP[gold_label_str]

                # Compute signals
                if self.evidence_mode == "oracle_evidence":
                    evidence_text = c.get("evidence") or ""
                    candidates = [(None, 1.0)] if evidence_text else []
                else:
                    candidates = self.retriever.retrieve(claim_text, top_k=self.top_k)
                    candidates = [(artifact, score) for artifact, score in candidates if score >= self.min_relevance_threshold]

                evidence_signals = []
                if self.evidence_mode == "oracle_evidence" and evidence_text:
                    evidence_signals.append((evidence_text, 1.0, None))
                else:
                    evidence_signals.extend(
                        (artifact.content.statement, score, artifact) for artifact, score in candidates
                    )

                has_evidence = bool(evidence_signals)
                if has_evidence:
                    scored = [
                        (
                            self.semantic_engine.compute_similarity(claim_text, text),
                            self.nli_engine.predict(text, claim_text),
                            score,
                            text,
                            artifact,
                        )
                        for text, score, artifact in evidence_signals
                    ]
                    sim = max(item[0] for item in scored)
                    nli_probs = {
                        key: max(item[1][key] for item in scored)
                        for key in ("entailment", "neutral", "contradiction")
                    }
                    retrieval_score = max(item[2] for item in scored)
                    evidence_text = scored[0][3]
                    evidence_count_normalized = min(1.0, len(scored) / max(self.top_k, 1))
                    support_present = any(item[1]["entailment"] >= 0.60 for item in scored)
                    contradiction_present = any(item[1]["contradiction"] >= 0.60 for item in scored)
                    evidence_disagreement = float(support_present and contradiction_present)
                    sources = [item[4].source.name for item in scored if item[4] is not None]
                    source_diversity = len(set(sources)) / len(scored) if sources else 0.0
                    mean_entailment = float(np.mean([item[1]["entailment"] for item in scored]))
                    mean_contradiction = float(np.mean([item[1]["contradiction"] for item in scored]))
                else:
                    sim = 0.05
                    nli_probs = {"entailment": 0.05, "neutral": 0.85, "contradiction": 0.10}
                    retrieval_score = 0.0
                    evidence_count_normalized = 0.0
                    evidence_disagreement = 0.0
                    source_diversity = 0.0
                    mean_entailment = 0.0
                    mean_contradiction = 0.0

                if self.evidence_mode == "retrieval" and candidates:
                    artifact, _ = candidates[0]
                    prov = ProvenanceAnalyzer().evaluate(artifact)
                    fresh = FreshnessAnalyzer().evaluate(claim_text, artifact)
                    source_rel = prov["source_reliability"]
                    prov_conf = prov["provenance_confidence"]
                    freshness_state = fresh["freshness_state"]
                    is_stale = fresh["is_stale"]
                    freshness_conf = fresh["freshness_confidence"]
                else:
                    prov = c.get("provenance", {})
                    source_rel = 0.95 if prov.get("source_type") == "primary" else 0.50
                    prov_conf = 0.92 if prov.get("verified") else 0.40
                    freshness_state = c.get("freshness", "CURRENT")
                    is_stale = (freshness_state == "STALE")
                    freshness_conf = 0.95 if freshness_state in ("CURRENT", "HISTORICAL") else 0.40

                ext_conf = 0.95

                feat_vec = self.feature_extractor.extract_vector(
                    semantic_similarity=sim,
                    nli_probs=nli_probs,
                    evidence_found=has_evidence,
                    retrieval_score=retrieval_score,
                    source_reliability=source_rel,
                    provenance_confidence=prov_conf,
                    freshness_confidence=freshness_conf,
                    is_stale=is_stale,
                    extraction_confidence=ext_conf,
                    evidence_disagreement=evidence_disagreement,
                    evidence_count_normalized=evidence_count_normalized,
                    source_diversity=source_diversity,
                    mean_nli_entailment=mean_entailment,
                    mean_nli_contradiction=mean_contradiction,
                )

                # Continuous risk: 0.0 for SUPPORTED, 1.0 for CONTRADICTED, 0.50 for UNVERIFIED
                if label_idx == 0:
                    continuous_risk = 0.05
                elif label_idx == 1:
                    continuous_risk = 0.95
                else:
                    continuous_risk = 0.50

                features_list.append(feat_vec)
                labels_list.append(label_idx)
                risk_list.append(continuous_risk)
                metadata_list.append({**c, "scenario_id": str(ex.get("example_id", ""))})
                group_list.append(str(ex.get("example_id", f"scenario-{len(group_list)}")))

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)
        risk = np.array(risk_list, dtype=np.float32)

        groups = np.asarray(group_list)
        # Group by source scenario so claims from the same QA example cannot cross splits.
        temp_ratio = val_ratio + test_ratio
        outer_split = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=self.random_seed)
        train_idx, temp_idx = next(outer_split.split(X, y, groups))
        X_train, X_temp = X[train_idx], X[temp_idx]
        y_train, y_temp = y[train_idx], y[temp_idx]
        risk_train, risk_temp = risk[train_idx], risk[temp_idx]
        meta_train = [metadata_list[i] for i in train_idx]
        meta_temp = [metadata_list[i] for i in temp_idx]
        groups_temp = groups[temp_idx]

        val_relative_ratio = val_ratio / temp_ratio
        inner_split = GroupShuffleSplit(
            n_splits=1, test_size=(1.0 - val_relative_ratio), random_state=self.random_seed
        )
        val_rel_idx, test_rel_idx = next(inner_split.split(X_temp, y_temp, groups_temp))
        X_val, X_test = X_temp[val_rel_idx], X_temp[test_rel_idx]
        y_val, y_test = y_temp[val_rel_idx], y_temp[test_rel_idx]
        risk_val, risk_test = risk_temp[val_rel_idx], risk_temp[test_rel_idx]
        meta_val = [meta_temp[i] for i in val_rel_idx]
        meta_test = [meta_temp[i] for i in test_rel_idx]

        # Fit feature scaler ONLY on training data (Zero Data Leakage)
        self.scaler.fit(X_train)
        X_train_norm = self.scaler.transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)

        # Compute balanced class weights for loss function
        class_counts = np.bincount(y_train, minlength=3)
        total_samples = len(y_train)
        class_weights = total_samples / (len(class_counts) * np.maximum(class_counts, 1).astype(np.float32))

        return {
            "splits": {
                "train": (X_train_norm, y_train, risk_train, meta_train),
                "val": (X_val_norm, y_val, risk_val, meta_val),
                "test": (X_test_norm, y_test, risk_test, meta_test),
            },
            "unscaled_splits": {
                "train": X_train,
                "val": X_val,
                "test": X_test,
            },
            "scaler": self.scaler,
            "class_weights": torch.from_numpy(class_weights.astype(np.float32)),
            "class_counts": class_counts.tolist(),
            "evidence_mode": self.evidence_mode,
            "nli_mode": self.nli_engine.mode,
            "dense_mode": self.retriever.dense_retriever.mode,
            "split_grouping": "scenario_id",
            "split_group_ids": {
                "train": sorted(set(groups[train_idx].tolist())),
                "validation": sorted(set(groups_temp[val_rel_idx].tolist())),
                "test": sorted(set(groups_temp[test_rel_idx].tolist())),
            },
        }

    def create_dataloaders(
        self, split_data: Dict[str, Any], batch_size: int = 16
    ) -> Dict[str, DataLoader]:
        dataloaders = {}
        for split_name in ["train", "val", "test"]:
            X, y, risk, _ = split_data["splits"][split_name]
            dataset = HallucinationDataset(X, y, risk)
            shuffle = (split_name == "train")
            dataloaders[split_name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )
        return dataloaders
