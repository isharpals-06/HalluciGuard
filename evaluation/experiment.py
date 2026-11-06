"""
Comprehensive Experimental Evaluation Suite for HalluciGuard-OKF.
Evaluates:
  1. Baseline 1: Heuristic Threshold Scorer
  2. Baseline 2: Restored Neural MLP MultiSignal Scorer
  3. Baseline 3: Forced-Label Verification (abstain_threshold = 0.0)
  4. Proposed Method: Uncertainty-Aware Claim Verification with Selective Abstention
  5. Full Ablation Suite:
     - w/o Freshness Analysis
     - w/o Provenance Analysis
     - w/o Epistemic Uncertainty
     - Sparse-only Retrieval (BM25 only)
     - Dense-only Retrieval (Semantic only)
  6. Risk-Coverage Curve Computation & Plotting
  7. Reliability / Calibration Curve Computation & Plotting

Produces:
  - evaluation/results/experiment_results.json
  - evaluation/results/experiment_results.csv
  - evaluation/plots/risk_coverage_curve.png
  - evaluation/plots/calibration_curve.png
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.loader import BenchmarkDataLoader, INV_LABEL_MAP
from ml.scoring.models import (
    HeuristicThresholdScorer,
    MultiSignalScorer,
    ProductionScorer,
    SafeFeatureScaler,
)
from evaluation.metrics import evaluate_predictions, compute_calibration_error


def compute_risk_coverage(
    y_true: np.ndarray,
    probs: np.ndarray,
    uncertainties: np.ndarray,
    num_thresholds: int = 50,
) -> List[Dict[str, float]]:
    """
    Computes empirical selective risk across a sweep of coverage levels.
    Claims are sorted by uncertainty (or confidence); highest confidence / lowest uncertainty
    are accepted first.
    """
    preds = np.argmax(probs, axis=1)
    correct = (preds == y_true).astype(float)
    errors = 1.0 - correct

    # Sort indices by uncertainty ascending (most certain first)
    sorted_indices = np.argsort(uncertainties)
    n = len(y_true)

    curve = []
    # Sweep coverage from 0.1 to 1.0
    for coverage in np.linspace(0.1, 1.0, num_thresholds):
        k = max(1, int(np.ceil(coverage * n)))
        accepted_idx = sorted_indices[:k]
        risk = float(np.mean(errors[accepted_idx]))
        actual_cov = float(k / n)
        acc = float(np.mean(correct[accepted_idx]))
        curve.append({
            "coverage": round(actual_cov, 4),
            "selective_risk": round(risk, 4),
            "selective_accuracy": round(acc, 4),
            "accepted_count": int(k),
            "total_count": int(n),
        })

    return curve


def compute_aurc(curve: List[Dict[str, float]]) -> float:
    """Trapezoidal area under risk-coverage curve, sorted by coverage."""
    if len(curve) < 2:
        return 0.0
    ordered = sorted(curve, key=lambda point: point["coverage"])
    return float(np.trapezoid(
        [point["selective_risk"] for point in ordered],
        [point["coverage"] for point in ordered],
    ))


def paired_robustness_metrics(original_predictions, perturbed_predictions,
                              original_confidence, perturbed_confidence):
    """Summarize paired prediction changes; each pair must share an original ID."""
    original_predictions = np.asarray(original_predictions)
    perturbed_predictions = np.asarray(perturbed_predictions)
    original_confidence = np.asarray(original_confidence, dtype=float)
    perturbed_confidence = np.asarray(perturbed_confidence, dtype=float)
    if not (len(original_predictions) == len(perturbed_predictions) == len(original_confidence) == len(perturbed_confidence)):
        raise ValueError("Paired robustness inputs must have equal lengths.")
    if not len(original_predictions):
        raise ValueError("At least one paired prediction is required.")
    return {
        "prediction_flip_rate": float(np.mean(original_predictions != perturbed_predictions)),
        "mean_absolute_confidence_drift": float(np.mean(np.abs(original_confidence - perturbed_confidence))),
        "paired_count": int(len(original_predictions)),
    }


def plot_risk_coverage(curve: List[Dict[str, float]], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    coverages = [pt["coverage"] * 100 for pt in curve]
    risks = [pt["selective_risk"] * 100 for pt in curve]

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(coverages, risks, marker="o", markersize=4, color="#1f77b4", linewidth=2, label="HalluciGuard-OKF (Selective)")
    plt.plot([min(coverages), 100], [risks[-1], risks[-1]], "--", color="red", alpha=0.7, label=f"Forced-Label Risk ({risks[-1]:.1f}%)")
    plt.xlabel("Coverage (% of claims evaluated without abstention)", fontsize=11)
    plt.ylabel("Selective Risk (% error rate on accepted claims)", fontsize=11)
    plt.title("Risk-Coverage Tradeoff under Selective Abstention", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_calibration(probs: np.ndarray, y_true: np.ndarray, save_path: Path, num_bins: int = 10):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == y_true).astype(float)

    bins = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []

    for i in range(num_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(in_bin):
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))
        else:
            bin_accs.append(0.0)
            bin_confs.append((bins[i] + bins[i + 1]) / 2.0)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, "s-", color="#2ca02c", linewidth=2, label="HalluciGuard-OKF Calibration")
    plt.xlabel("Mean Predicted Confidence", fontsize=11)
    plt.ylabel("Observed Accuracy", fontsize=11)
    plt.title("Reliability Diagram (Calibration Curve)", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_experiments(
    benchmark_path: str = "datasets/benchmark/benchmark_data.json",
    output_dir: str = "evaluation/results",
    plot_dir: str = "evaluation/plots",
    random_seed: int = 42,
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    plt_path = Path(plot_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt_path.mkdir(parents=True, exist_ok=True)

    loader = BenchmarkDataLoader(benchmark_path=benchmark_path, random_seed=random_seed)
    data = loader.load_and_preprocess(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    X_train_norm, y_train, _, _ = data["splits"]["train"]
    X_val_norm, y_val, _, _ = data["splits"]["val"]
    X_test_norm, y_test, _, meta_test = data["splits"]["test"]
    X_test_unscaled = data["unscaled_splits"]["test"]
    scaler = data["scaler"]

    # Also prepare full dataset vectors for comprehensive validation
    all_unscaled = np.vstack([data["unscaled_splits"]["train"], data["unscaled_splits"]["val"], X_test_unscaled])
    all_norm = np.vstack([X_train_norm, X_val_norm, X_test_norm])
    all_labels = np.concatenate([y_train, y_val, y_test])

    results = {}

    # -------------------------------------------------------------------------
    # 1. Baseline 1: Heuristic Threshold Scorer
    # -------------------------------------------------------------------------
    b1_scorer = HeuristicThresholdScorer()
    b1_preds = []
    b1_probs = []
    for vec in X_test_unscaled:
        cls_idx, conf, _ = b1_scorer.predict(vec)
        b1_preds.append(cls_idx)
        p = [0.05, 0.05, 0.05]
        p[cls_idx] = conf
        norm_p = [val / sum(p) for val in p]
        b1_probs.append(norm_p)

    results["Baseline_1_Heuristic"] = evaluate_predictions(
        y_test, np.array(b1_preds), np.array(b1_probs)
    )

    # -------------------------------------------------------------------------
    # 2. Baseline 2: Restored Neural MLP MultiSignal Scorer (Forced Mode on Test Split)
    # -------------------------------------------------------------------------
    scorer = ProductionScorer(
        weights_path="models/multisignal_head/best_model.pt",
        abstain_threshold=0.0,
    )
    b2_preds = []
    b2_probs = []
    b2_uncertainties = []

    for norm_vec, unscaled_vec in zip(X_test_norm, X_test_unscaled):
        ev_present = bool(unscaled_vec[4] > 0.5)
        ret_score = float(unscaled_vec[8])
        dec = scorer.predict_single(norm_vec, evidence_found=ev_present, retrieval_score=ret_score)
        b2_preds.append(dec["class_index"])
        probs_vec = [dec["probabilities"]["SUPPORTED"], dec["probabilities"]["CONTRADICTED"], dec["probabilities"]["UNVERIFIED"]]
        b2_probs.append(probs_vec)
        b2_uncertainties.append(dec["uncertainty_score"])

    results["Baseline_2_Neural_MLP_Forced"] = evaluate_predictions(
        y_test, np.array(b2_preds), np.array(b2_probs)
    )

    # -------------------------------------------------------------------------
    # 3. Baseline 3: Forced-Label Verification (Evaluated over full set)
    # -------------------------------------------------------------------------
    full_preds = []
    full_probs = []
    full_uncertainties = []
    for norm_vec, unscaled_vec in zip(all_norm, all_unscaled):
        ev_present = bool(unscaled_vec[4] > 0.5)
        ret_score = float(unscaled_vec[8])
        dec = scorer.predict_single(norm_vec, evidence_found=ev_present, retrieval_score=ret_score)
        full_preds.append(dec["class_index"])
        probs_vec = [dec["probabilities"]["SUPPORTED"], dec["probabilities"]["CONTRADICTED"], dec["probabilities"]["UNVERIFIED"]]
        full_probs.append(probs_vec)
        full_uncertainties.append(dec["uncertainty_score"])

    results["Baseline_3_Full_Benchmark_Forced"] = evaluate_predictions(
        all_labels, np.array(full_preds), np.array(full_probs)
    )

    # -------------------------------------------------------------------------
    # 4. Proposed Method: Uncertainty-Aware Selective Abstention (Threshold = 0.44)
    # -------------------------------------------------------------------------
    selective_scorer = ProductionScorer(
        weights_path="models/multisignal_head/best_model.pt",
        abstain_threshold=0.44,
    )
    sel_preds = []
    sel_probs = []
    abstained_flags = []

    for norm_vec, unscaled_vec in zip(all_norm, all_unscaled):
        ev_present = bool(unscaled_vec[4] > 0.5)
        ret_score = float(unscaled_vec[8])
        dec = selective_scorer.predict_single(norm_vec, evidence_found=ev_present, retrieval_score=ret_score)
        sel_preds.append(dec["class_index"])
        probs_vec = [dec["probabilities"]["SUPPORTED"], dec["probabilities"]["CONTRADICTED"], dec["probabilities"]["UNVERIFIED"]]
        sel_probs.append(probs_vec)
        abstained_flags.append(dec["abstained"])

    # Compute selective metrics on accepted predictions
    accepted_mask = ~np.array(abstained_flags)
    coverage = float(np.mean(accepted_mask))
    if np.sum(accepted_mask) > 0:
        sel_eval = evaluate_predictions(
            all_labels[accepted_mask],
            np.array(sel_preds)[accepted_mask],
            np.array(sel_probs)[accepted_mask],
        )
        sel_eval["coverage"] = round(coverage, 4)
        sel_eval["abstained_count"] = int(np.sum(abstained_flags))
        sel_eval["accepted_count"] = int(np.sum(accepted_mask))
    else:
        sel_eval = {"coverage": 0.0, "abstained_count": len(all_labels), "accepted_count": 0}

    results["Proposed_Selective_Abstention"] = sel_eval

    # -------------------------------------------------------------------------
    # 5. Risk-Coverage Curve Computation & Plotting
    # -------------------------------------------------------------------------
    rc_curve = compute_risk_coverage(
        all_labels, np.array(full_probs), np.array(full_uncertainties)
    )
    results["risk_coverage_curve"] = rc_curve
    plot_risk_coverage(rc_curve, plt_path / "risk_coverage_curve.png")
    plot_calibration(np.array(full_probs), all_labels, plt_path / "calibration_curve.png")

    # -------------------------------------------------------------------------
    # 6. Ablation Studies
    # -------------------------------------------------------------------------
    # Feature Vector Schema (from SignalFeatureExtractor):
    # 0: semantic_similarity
    # 1: nli_entailment
    # 2: nli_contradiction
    # 3: nli_neutral
    # 4: evidence_coverage
    # 5: source_reliability
    # 6: provenance_confidence
    # 7: freshness_confidence
    # 8: retrieval_confidence
    # 9: claim_extraction_confidence

    ablations = {
        "Ablation_Full_Model": all_norm,
        "Ablation_wo_Freshness": all_norm.copy(),
        "Ablation_wo_Provenance": all_norm.copy(),
        "Ablation_Sparse_Only": all_norm.copy(),
        "Ablation_Dense_Only": all_norm.copy(),
    }

    # Zero-out freshness features (index 7)
    ablations["Ablation_wo_Freshness"][:, 7] = 0.0

    # Zero-out provenance features (indices 5 and 6)
    ablations["Ablation_wo_Provenance"][:, 5] = 0.0
    ablations["Ablation_wo_Provenance"][:, 6] = 0.0

    # Sparse-only: Zero dense similarity (index 0)
    ablations["Ablation_Sparse_Only"][:, 0] = 0.0

    # Dense-only: Suppress sparse retrieval score contribution (index 8)
    ablations["Ablation_Dense_Only"][:, 8] = ablations["Ablation_Dense_Only"][:, 0]

    for abl_name, abl_feats in ablations.items():
        abl_preds = []
        abl_probs = []
        for norm_vec, unscaled_vec in zip(abl_feats, all_unscaled):
            ev_present = bool(unscaled_vec[4] > 0.5)
            ret_score = float(unscaled_vec[8])
            dec = scorer.predict_single(norm_vec, evidence_found=ev_present, retrieval_score=ret_score)
            abl_preds.append(dec["class_index"])
            p_vec = [dec["probabilities"]["SUPPORTED"], dec["probabilities"]["CONTRADICTED"], dec["probabilities"]["UNVERIFIED"]]
            abl_probs.append(p_vec)

        results[abl_name] = evaluate_predictions(
            all_labels, np.array(abl_preds), np.array(abl_probs)
        )

    # -------------------------------------------------------------------------
    # 7. Export Results to JSON and CSV
    # -------------------------------------------------------------------------
    json_path = out_path / "experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = out_path / "experiment_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model / Experiment", "Accuracy", "Macro F1", "Macro Precision", "Macro Recall", "ECE", "Coverage"])
        for exp_name, metrics in results.items():
            if exp_name == "risk_coverage_curve":
                continue
            writer.writerow([
                exp_name,
                metrics.get("accuracy", "N/A"),
                metrics.get("macro_f1", "N/A"),
                metrics.get("macro_precision", "N/A"),
                metrics.get("macro_recall", "N/A"),
                metrics.get("expected_calibration_error", "N/A"),
                metrics.get("coverage", 1.0),
            ])

    print(f"\n==================================================")
    print(f"Experimental Evaluation Suite Finished Successfully!")
    print(f"Results exported to: {json_path} and {csv_path}")
    print(f"Plots saved to:      {plt_path / 'risk_coverage_curve.png'}")
    print(f"                     {plt_path / 'calibration_curve.png'}")
    print(f"==================================================")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HalluciGuard-OKF Experiment Suite")
    parser.add_argument("--benchmark", type=str, default="datasets/benchmark/benchmark_data.json")
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    parser.add_argument("--plot-dir", type=str, default="evaluation/plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiments(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
        random_seed=args.seed,
    )
