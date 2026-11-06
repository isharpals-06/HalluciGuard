"""Validation-only post-hoc temperature scaling for multiclass logits."""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """Fit one positive temperature by minimizing validation negative log-likelihood."""
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        raise ValueError("Temperature scaling requires at least two validation classes.")
    x = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.5, 5.0)
        loss = F.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.5, 5.0).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x = torch.as_tensor(logits, dtype=torch.float64)
    return torch.softmax(x / temperature, dim=-1).cpu().numpy()


def select_abstention_threshold(probabilities: np.ndarray, labels: np.ndarray,
                                target_risk: float = 0.10) -> Tuple[float, dict]:
    """Choose the highest-coverage confidence threshold meeting validation risk.

    Returns an infinite threshold when no non-empty accepted set satisfies the
    risk constraint. This is a validation policy selector, not a test tuner.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probabilities.ndim != 2 or len(probabilities) != len(labels) or not len(labels):
        raise ValueError("Validation probabilities and labels must be non-empty and aligned.")
    if not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be in [0, 1].")
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    candidates = sorted(set([0.0, *confidence.tolist()]))
    valid = []
    for threshold in candidates:
        accepted = confidence >= threshold
        if not accepted.any():
            continue
        risk = float(np.mean(predictions[accepted] != labels[accepted]))
        coverage = float(np.mean(accepted))
        if risk <= target_risk:
            valid.append((coverage, threshold, risk, int(accepted.sum())))
    if not valid:
        return 1.000001, {"coverage": 0.0, "risk": None, "accepted_count": 0, "target_risk": target_risk}
    coverage, threshold, risk, accepted_count = max(valid, key=lambda item: (item[0], -item[1]))
    return float(threshold), {
        "coverage": coverage, "risk": risk, "accepted_count": accepted_count,
        "target_risk": target_risk,
    }
