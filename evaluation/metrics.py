"""
Evaluation Metrics Framework for HalluciGuard-OKF.
Computes multi-class classification metrics, calibration error (ECE), and confusion matrices.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    average_precision_score,
)


def compute_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class predictions.
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics.
    """
    if class_names is None:
        class_names = ["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("probabilities must be an N x number-of-classes matrix")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (row_sums <= 0).any():
        raise ValueError("probabilities must contain finite, non-negative values with positive row sums")
    # Renormalize to absorb floating-point drift and heuristic score vectors.
    probabilities = probabilities / row_sums

    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_support = (
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    ece = compute_calibration_error(probabilities, y_true)
    one_hot = np.eye(len(class_names))[np.asarray(y_true, dtype=int)]
    multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    nll = float(log_loss(y_true, probabilities, labels=list(range(len(class_names)))))
    try:
        pr_auc = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        pr_auc = 0.0

    # Per-class breakdown
    class_metrics = {}
    for i, name in enumerate(class_names):
        class_metrics[name] = {
            "precision": round(float(per_class_p[i]), 4) if i < len(per_class_p) else 0.0,
            "recall": round(float(per_class_r[i]), 4) if i < len(per_class_r) else 0.0,
            "f1": round(float(per_class_f1[i]), 4) if i < len(per_class_f1) else 0.0,
            "support": int(per_class_support[i]) if i < len(per_class_support) else 0,
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "expected_calibration_error": round(float(ece), 4),
        "brier_score": round(multiclass_brier, 4),
        "negative_log_likelihood": round(nll, 4),
        "macro_pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": class_metrics,
    }
