"""
Production Training Pipeline for HalluciGuard-OKF Multi-Signal Head.
Executes training, validation, early stopping, checkpointing, baseline comparison, and evaluation.
Usage:
    python train.py --config configs/train.yaml
"""

from typing import Tuple, Dict, Any, Optional
import argparse
import json
import os
import random
import sys
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from sklearn.linear_model import LogisticRegression
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from datasets.loader import BenchmarkDataLoader, INV_LABEL_MAP
from ml.scoring.models import MultiSignalScorer, HeuristicThresholdScorer
from evaluation.metrics import evaluate_predictions
from evaluation.calibration import fit_temperature, select_abstention_threshold
from evaluation.experiment import compute_risk_coverage, compute_aurc, plot_risk_coverage, plot_calibration


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for features, labels, risk in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        risk = risk.to(device)

        optimizer.zero_grad()
        logits, pred_risk = model(features)

        loss_cls = cls_criterion(logits, labels)
        loss_reg = reg_criterion(pred_risk, risk)
        loss = loss_cls + 0.5 * loss_reg

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_cls_loss += loss_cls.item() * len(labels)
        total_reg_loss += loss_reg.item() * len(labels)

    n = len(dataloader.dataset)
    return total_cls_loss / n, total_reg_loss / n


def evaluate_split(
    model: nn.Module, dataloader, device: torch.device, temperature: float = 1.0
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for features, labels, _ in dataloader:
            features = features.to(device)
            logits, _ = model(features)
            probs = torch.softmax(logits / temperature, dim=-1)

            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_arr = np.array(all_probs)

    metrics = evaluate_predictions(y_true, y_pred, probs_arr)
    return metrics, y_true, y_pred, probs_arr


def run_training(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_config_path = config["data"].get("model_config_path", "configs/model_config.yaml")
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    seed = config["training"].get("seed", config["experiment"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cpu")

    print(f"==================================================")
    print(f"Starting HalluciGuard-OKF Training Run: {config['experiment']['name']}")
    print(f"Random Seed: {seed} | Device: {device}")
    print(f"==================================================")

    # 1. Load Data
    data_loader = BenchmarkDataLoader(
        benchmark_path=config["data"]["benchmark_path"],
        random_seed=seed,
        evidence_mode=config["data"].get("evidence_mode", "retrieval"),
        knowledge_path=config["data"].get("knowledge_path", "knowledge/artifacts/curated_knowledge.json"),
        nli_mode=model_config.get("nli", {}).get("mode", "rule_based"),
        nli_model_name=model_config.get("nli", {}).get("model_name", "roberta-large-mnli"),
        dense_mode=model_config.get("embedding", {}).get("mode", "tfidf"),
        top_k=model_config.get("retrieval", {}).get("top_k", 5),
        min_relevance_threshold=model_config.get("retrieval", {}).get("min_relevance_threshold", 0.35),
    )
    data = data_loader.load_and_preprocess(
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
    )

    batch_size = config["training"]["batch_size"]
    dataloaders = data_loader.create_dataloaders(data, batch_size=batch_size)

    class_weights = data["class_weights"].to(device)
    print(f"Class distribution in train split: {data['class_counts']}")
    print(f"Balanced class weights: {class_weights.cpu().numpy().round(3)}")

    # 2. Evaluate Baseline Model (Phase 8) on unscaled test split
    print("\n--- Evaluating Baseline Model (Experiment A/B/C Baseline) ---")
    baseline = HeuristicThresholdScorer()
    X_test_unscaled = data["unscaled_splits"]["test"]
    _, y_test, _, meta_test = data["splits"]["test"]

    base_preds = []
    base_probs = []
    for vec in X_test_unscaled:
        pred_cls, conf, _ = baseline.predict(vec)
        base_preds.append(pred_cls)
        # Formulate pseudo-probability vector
        p = [0.05, 0.05, 0.05]
        p[pred_cls] = conf
        norm_p = [val / sum(p) for val in p]
        base_probs.append(norm_p)

    baseline_metrics = evaluate_predictions(
        y_test, np.array(base_preds), np.array(base_probs)
    )

    # Lightweight learned baseline, fit only on the training partition.
    X_train, y_train, _, _ = data["splits"]["train"]
    linear_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    linear_model.fit(X_train, y_train)
    linear_probs = linear_model.predict_proba(data["splits"]["test"][0])
    linear_preds = linear_probs.argmax(axis=1)
    logistic_metrics = evaluate_predictions(y_test, linear_preds, linear_probs)

    # Frozen rule-based NLI-only baseline; thresholds are explicit and its
    # confidence vector is a scoring heuristic, not calibrated probability.
    nli_preds, nli_probs = [], []
    for vec in X_test_unscaled:
        entailment, contradiction, evidence_found = float(vec[1]), float(vec[2]), bool(vec[4] >= 0.5)
        if not evidence_found:
            pred, score = 2, 0.85
        elif contradiction >= 0.60 and contradiction >= entailment:
            pred, score = 1, contradiction
        elif entailment >= 0.60:
            pred, score = 0, entailment
        else:
            pred, score = 2, max(0.50, float(vec[3]))
        row = np.full(3, 0.05, dtype=float)
        row[pred] = max(0.05, score)
        nli_preds.append(pred)
        nli_probs.append(row / row.sum())
    nli_only_metrics = evaluate_predictions(y_test, np.asarray(nli_preds), np.asarray(nli_probs))
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"Baseline Macro F1: {baseline_metrics['macro_f1']:.4f}")
    print(f"Baseline ECE:      {baseline_metrics['expected_calibration_error']:.4f}")

    # 3. Initialize Proposed Model (Phase 3)
    model = MultiSignalScorer(
        input_dim=config["model"]["input_dim"],
        hidden_dims=tuple(config["model"]["hidden_dims"]),
        dropout=config["model"]["dropout"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    reg_criterion = nn.MSELoss()

    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Training Loop with Early Stopping
    out_dir = Path(config["artifacts"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / config["artifacts"]["checkpoint_name"]

    patience = config["training"]["patience"]
    patience_counter = 0
    best_val_f1 = -1.0
    history = []

    print("\n--- Training Proposed Multi-Signal Architecture ---")
    for epoch in range(1, epochs + 1):
        cls_loss, reg_loss = train_epoch(
            model, dataloaders["train"], optimizer, cls_criterion, reg_criterion, device
        )
        scheduler.step()

        val_metrics, _, _, _ = evaluate_split(model, dataloaders["val"], device)
        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        history.append({
            "epoch": epoch,
            "cls_loss": round(cls_loss, 4),
            "reg_loss": round(reg_loss, 4),
            "val_f1": val_f1,
            "val_acc": val_acc,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Cls Loss: {cls_loss:.4f} | "
                f"Reg Loss: {reg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

        if val_f1 > best_val_f1 + config["training"]["min_delta"]:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            if SAFETENSORS_AVAILABLE:
                safetensors.torch.save_file(model.state_dict(), str(out_dir / "best_model.safetensors"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
                break

    # 5. Final Evaluation on Held-out Isolated Test Set
    print("\n--- Final Evaluation on Isolated Test Set ---")
    # Load best checkpoint safely
    if SAFETENSORS_AVAILABLE and (out_dir / "best_model.safetensors").exists():
        state_dict = safetensors.torch.load_file(str(out_dir / "best_model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    validation_logits = []
    validation_labels = []
    model.eval()
    with torch.no_grad():
        for features, labels, _ in dataloaders["val"]:
            logits, _ = model(features.to(device))
            validation_logits.extend(logits.cpu().numpy())
            validation_labels.extend(labels.numpy())
    val_metrics_precal, _, _, _ = evaluate_split(model, dataloaders["val"], device)
    calibration_temperature = fit_temperature(
        np.asarray(validation_logits), np.asarray(validation_labels)
    )
    val_metrics_postcal, y_true_val, y_pred_val, probs_val = evaluate_split(
        model, dataloaders["val"], device, temperature=calibration_temperature
    )
    print(f"Validation-fitted temperature: {calibration_temperature:.4f}")
    test_metrics, y_true_test, y_pred_test, probs_test = evaluate_split(
        model, dataloaders["test"], device, temperature=calibration_temperature
    )
    prediction_path = Path("evaluation/results/current_training_run.json")
    abstention_threshold, validation_selection = select_abstention_threshold(
        probs_val, y_true_val, target_risk=float(config["training"].get("target_selective_risk", 0.10))
    )
    test_confidence = probs_test.max(axis=1)
    test_accepted = test_confidence >= abstention_threshold
    test_errors = y_pred_test != y_true_test
    selective_metrics = {
        "threshold_selected_on_validation": abstention_threshold,
        "coverage": float(np.mean(test_accepted)),
        "accepted_count": int(test_accepted.sum()),
        "total_count": int(len(test_accepted)),
        "selective_risk": float(np.mean(test_errors[test_accepted])) if test_accepted.any() else None,
        "accuracy": float(np.mean(y_pred_test[test_accepted] == y_true_test[test_accepted]))
        if test_accepted.any() else None,
        "accepted_count_by_class": {
            INV_LABEL_MAP[class_id]: int(np.sum(test_accepted & (y_true_test == class_id)))
            for class_id in range(3)
        },
        "accepted_coverage_by_class": {
            INV_LABEL_MAP[class_id]: float(np.mean(test_accepted[y_true_test == class_id]))
            if np.any(y_true_test == class_id) else None
            for class_id in range(3)
        },
    }
    risk_curve = compute_risk_coverage(y_true_test, probs_test, 1.0 - test_confidence)
    selective_metrics["aurc"] = compute_aurc(risk_curve)
    plot_dir = Path("evaluation/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_risk_coverage(risk_curve, plot_dir / "risk_coverage_curve.png")
    plot_calibration(probs_test, y_true_test, plot_dir / "calibration_curve.png")

    print(f"Proposed Model Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Proposed Model Test Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"Proposed Model Test ECE:       {test_metrics['expected_calibration_error']:.4f}")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")

    # 6. Error Analysis (Phase 10)
    error_analysis_cases = []
    for i in range(len(y_true_test)):
        true_cls = int(y_true_test[i])
        pred_cls = int(y_pred_test[i])
        if true_cls != pred_cls:
            meta = meta_test[i]
            error_analysis_cases.append({
                "claim_id": meta.get("claim_id", f"Test-{i}"),
                "claim_text": meta.get("claim_text"),
                "true_label": INV_LABEL_MAP[true_cls],
                "predicted_label": INV_LABEL_MAP[pred_cls],
                "confidence": float(np.max(probs_test[i])),
                "predicted_probs": {
                    INV_LABEL_MAP[k]: round(float(probs_test[i][k]), 4) for k in range(3)
                },
                "error_type": "FALSE_POSITIVE" if pred_cls == 1 else "FALSE_NEGATIVE" if pred_cls == 0 else "INSUFFICIENT_EVIDENCE",
                "hallucination_pattern": meta.get("hallucination_pattern", "unknown"),
            })

    # 7. Save Artifacts & Scaler (Phase 13)
    scaler_path = out_dir / config["artifacts"]["scaler_name"]
    # Save safe JSON feature scaler
    scaler_json_path = out_dir / config["artifacts"].get("scaler_name", "feature_scaler.json")
    with open(scaler_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "mean": data["scaler"].mean_.tolist(),
            "scale": data["scaler"].scale_.tolist(),
            "var": data["scaler"].var_.tolist(),
            "n_features_in": int(data["scaler"].n_features_in_),
        }, f, indent=2)

    metadata = {
        "model_name": config["experiment"]["name"],
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "input_features": config["model"]["input_dim"],
        "classes": INV_LABEL_MAP,
        "baseline_comparison": {
            "heuristic_baseline": baseline_metrics,
            "nli_only_rule_baseline": nli_only_metrics,
            "logistic_regression_baseline": logistic_metrics,
            "proposed_model": test_metrics,
            "f1_improvement_vs_heuristic": round(test_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4),
            "accuracy_improvement_vs_heuristic": round(test_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
        },
        "test_metrics": test_metrics,
        "validation_metrics": val_metrics_postcal,
        "selective_metrics": selective_metrics,
        "prediction_artifacts_path": str(prediction_path),
        "error_analysis_samples": error_analysis_cases,
        "training_history": history,
        "reproducibility": {
            "dataset_path": str(Path(config["data"]["benchmark_path"])),
            "dataset_sha256": hashlib.sha256(Path(config["data"]["benchmark_path"]).read_bytes()).hexdigest(),
            "training_config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "model_config_path": model_config_path,
            "model_config_sha256": hashlib.sha256(Path(model_config_path).read_bytes()).hexdigest(),
            "source_fingerprint_sha256": hashlib.sha256(b"".join(
                Path(source_path).read_bytes() for source_path in [
                    "train.py", "datasets/loader.py", "inference.py", "retrieval/hybrid.py",
                    "ml/nli/nli_engine.py", "ml/scoring/features.py", "ml/scoring/models.py",
                    "evaluation/calibration.py", "evaluation/metrics.py", "evaluation/perturbations.py",
                ]
            )).hexdigest(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "nli_mode": data["nli_mode"],
            "dense_retrieval_mode": data["dense_mode"],
            "evidence_mode": data["evidence_mode"],
            "top_k": int(model_config.get("retrieval", {}).get("top_k", 5)),
            "min_relevance_threshold": float(model_config.get("retrieval", {}).get("min_relevance_threshold", 0.35)),
            "seed": seed,
            "split_ratios": {
                "train": config["data"]["train_split"],
                "validation": config["data"]["val_split"],
                "test": config["data"]["test_split"],
            },
            "split_grouping": data["split_grouping"],
            "split_group_ids": data["split_group_ids"],
            "calibration_method": "validation_fitted_temperature_scaling",
            "posthoc_calibration_method": "temperature_scaling",
            "posthoc_temperature": calibration_temperature,
            "calibration_validation_count": int(len(validation_labels)),
            "calibration_temperature_at_lower_bound": bool(calibration_temperature <= 0.50001),
            "calibration_temperature_bounds": [0.5, 5.0],
            "validation_metrics_before_calibration": val_metrics_precal,
            "validation_metrics_after_calibration": val_metrics_postcal,
            "abstention_threshold": abstention_threshold,
            "target_selective_risk": float(config["training"].get("target_selective_risk", 0.10)),
            "selected_abstention_threshold": abstention_threshold,
            "validation_selective_operating_point": validation_selection,
        },
    }

    meta_path = out_dir / config["artifacts"]["metadata_name"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    def prediction_records(split_name, truth, predictions, probabilities):
        split_metadata = data["splits"][split_name][3]
        return [{
            "claim_id": item.get("claim_id"),
            "scenario_id": item.get("scenario_id"),
            "gold_label": INV_LABEL_MAP[int(truth[i])],
            "predicted_label": INV_LABEL_MAP[int(predictions[i])],
            "probabilities": {
                INV_LABEL_MAP[class_id]: float(probabilities[i][class_id])
                for class_id in range(3)
            },
        } for i, item in enumerate(split_metadata)]

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_path, "w", encoding="utf-8") as result_file:
        json.dump({
            "metadata": metadata["reproducibility"],
            "baseline_comparison": metadata["baseline_comparison"],
            "validation_metrics": val_metrics_postcal,
            "test_metrics": test_metrics,
            "selective_metrics": selective_metrics,
            "risk_coverage_curve": risk_curve,
            "validation_predictions": prediction_records("val", y_true_val, y_pred_val, probs_val),
            "test_predictions": prediction_records("test", y_true_test, y_pred_test, probs_test),
        }, result_file, indent=2)

    print(f"\nTraining pipeline complete! Model artifacts saved to {out_dir}")
    print(
        "Macro F1 difference vs heuristic baseline: "
        f"{metadata['baseline_comparison']['f1_improvement_vs_heuristic']:+.4f}"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HalluciGuard-OKF Multi-Signal Head")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML config")
    args = parser.parse_args()

    run_training(args.config)
