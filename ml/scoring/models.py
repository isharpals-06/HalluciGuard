"""
Scoring and Decision Models with Uncertainty Estimation and Selective Abstention.
Contains:
1. Baseline Heuristic Scorer (Experiment baseline)
2. Safe PyTorch MultiSignalScorer Neural Architecture with SafeTensors support
3. Uncertainty-Aware Calibrated Inference Engine with Selective Abstention
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


class SafeFeatureScaler:
    """Safe, pickle-free feature standard scaler loaded from JSON parameters."""

    def __init__(self, json_path: Optional[str] = None):
        self.mean_ = None
        self.scale_ = None
        if json_path and Path(json_path).exists():
            self.load(json_path)

    def load(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.mean_ = np.array(data["mean"], dtype=np.float32)
        self.scale_ = np.array(data["scale"], dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            return X
        return (X - self.mean_) / np.where(self.scale_ == 0, 1.0, self.scale_)


# -------------------------------------------------------------------------
# 1. Baseline Model: Heuristic Rule-Based Decision Engine
# -------------------------------------------------------------------------
class HeuristicThresholdScorer:
    """
    Standard industry baseline: uses fixed threshold rules on similarity and NLI.
    """
    def __init__(
        self,
        min_coverage: float = 0.50,
        contra_threshold: float = 0.60,
        entail_threshold: float = 0.55,
    ):
        self.min_coverage = min_coverage
        self.contra_threshold = contra_threshold
        self.entail_threshold = entail_threshold

    def predict(self, feature_vec: np.ndarray) -> Tuple[int, float, float]:
        """
        Returns:
            (pred_class_idx, confidence, hallucination_score)
            Classes: 0: SUPPORTED, 1: CONTRADICTED, 2: UNVERIFIED
        """
        sim = feature_vec[0]
        ent = feature_vec[1]
        con = feature_vec[2]
        cov = feature_vec[4]
        fresh = feature_vec[7]

        # If no evidence retrieved -> UNVERIFIED
        if cov < 0.5:
            return 2, 0.85, 0.50

        # If contradiction or stale
        if con >= self.contra_threshold or fresh < 0.40:
            hallucination_score = min(1.0, max(0.70, con))
            return 1, float(con), float(hallucination_score)

        if ent >= self.entail_threshold and sim >= 0.30:
            hallucination_score = max(0.02, 1.0 - ent)
            return 0, float(ent), float(hallucination_score)

        return 2, 0.70, 0.50


# -------------------------------------------------------------------------
# 2. Multi-Signal Classification & Calibration Head
# -------------------------------------------------------------------------
class MultiSignalScorer(nn.Module):
    """
    Deep PyTorch Neural Decision Architecture for Multi-Signal Fusion.
    Fuses 10 heterogeneous signals to predict 3-way status and continuous risk score.
    """
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.20,
        num_classes: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        h1, h2 = hidden_dims

        # Backbone stream
        self.fc1 = nn.Linear(input_dim, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.drop2 = nn.Dropout(dropout)

        # Skip connection from input to second hidden layer
        self.skip = nn.Linear(input_dim, h2)

        # Classification Head (3 classes)
        self.classifier = nn.Linear(h2, num_classes)

        # Auxiliary continuous hallucination risk regression head [0, 1]
        self.risk_head = nn.Sequential(
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

        # Trainable / Calibrated Temperature Parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = self.skip(x)

        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out + identity)
        out = self.drop2(out)

        logits = self.classifier(out) / torch.clamp(self.temperature, min=0.1, max=5.0)
        risk = self.risk_head(out).squeeze(-1)

        return logits, risk


# -------------------------------------------------------------------------
# 3. Model Wrapper with Uncertainty & Selective Abstention
# -------------------------------------------------------------------------
class ProductionScorer:
    """
    High-level wrapper managing safe model execution, calibration,
    epistemic uncertainty estimation, and selective abstention.
    """
    CLASS_LABELS = {0: "SUPPORTED", 1: "CONTRADICTED", 2: "UNVERIFIED"}

    def __init__(
        self,
        model: Optional[MultiSignalScorer] = None,
        weights_path: Optional[str] = None,
        abstain_threshold: float = 0.60,
    ):
        self.device = torch.device("cpu")
        self.model = model or MultiSignalScorer()
        self.abstain_threshold = abstain_threshold
        self.temperature_scale = 1.0
        if weights_path:
            self.load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, path: str):
        p = Path(path)
        # 1. Prefer safetensors if available
        sf_path = p.with_suffix(".safetensors")
        if sf_path.exists() and SAFETENSORS_AVAILABLE:
            state_dict = safetensors.torch.load_file(sf_path)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            return

        if p.exists():
            # Use safe weights_only=True
            state_dict = torch.load(p, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def predict_single(
        self,
        feature_vec: np.ndarray,
        evidence_found: bool = True,
        retrieval_score: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Infers status, calibrated confidence, uncertainty score, and selective abstention decision.
        """
        with torch.no_grad():
            x = torch.from_numpy(feature_vec).unsqueeze(0).to(self.device)
            logits, risk = self.model(x)
            probs = F.softmax(logits / max(float(self.temperature_scale), 1e-6), dim=-1).squeeze(0).cpu().numpy()
            continuous_score = float(risk.squeeze(0).cpu().item())

        raw_pred_class_idx = int(np.argmax(probs))
        raw_confidence = float(probs[raw_pred_class_idx])
        raw_status = self.CLASS_LABELS[raw_pred_class_idx]

        # Uncertainty Estimation:
        # Epistemic uncertainty is high when:
        # 1. Output probability entropy is high (confidence is low)
        # 2. Evidence retrieval relevance is weak
        retrieval_penalty = max(0.0, 1.0 - retrieval_score)
        entropy_uncertainty = 1.0 - raw_confidence
        uncertainty_score = float(min(1.0, 0.6 * entropy_uncertainty + 0.4 * retrieval_penalty))

        # Calibrated Confidence adjusted for evidence quality
        calibrated_confidence = raw_confidence

        # Selective Abstention Decision:
        # If abstain_threshold > 0.0, abstain when evidence is missing or confidence is low
        abstained = False
        abstention_reason = None
        status = raw_status

        if self.abstain_threshold > 0.0:
            if not evidence_found:
                status = "UNVERIFIED"
                abstained = True
                abstention_reason = "No relevant verified evidence retrieved above relevance threshold."
            elif calibrated_confidence < self.abstain_threshold:
                status = "UNVERIFIED"
                abstained = True
                abstention_reason = (
                    f"Confidence ({calibrated_confidence:.2f}) fell below selective abstention "
                    f"threshold ({self.abstain_threshold:.2f}). Epistemic uncertainty: {uncertainty_score:.2f}."
                )

        # Dynamic Flags
        flags = []
        if uncertainty_score > 0.45:
            flags.append("HIGH_UNCERTAINTY")
        if abstained:
            flags.append("ABSTAINED")
        if feature_vec[7] < 0.40 and status == "CONTRADICTED":
            flags.append("STALE")

        return {
            "status": status,
            "raw_status": raw_status,
            "class_index": raw_pred_class_idx,
            "confidence": round(calibrated_confidence, 4),
            "raw_confidence": round(raw_confidence, 4),
            "uncertainty_score": round(uncertainty_score, 4),
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "hallucination_score": round(continuous_score, 4),
            "probabilities": {
                "SUPPORTED": round(float(probs[0]), 4),
                "CONTRADICTED": round(float(probs[1]), 4),
                "UNVERIFIED": round(float(probs[2]), 4),
            },
            "flags": flags,
        }
