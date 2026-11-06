"""
Script to build the comprehensive research and experimentation notebook
'notebooks/08_model_training.ipynb' containing all 21 sections requested in Phase 7.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell("""# HalluciGuard-OKF: Research, Experimentation, and Model Pipeline Notebook
> **Explainable Open Knowledge Format-Based LLM Hallucination Detection and Verification Framework**

This notebook serves as the interactive research and experimentation laboratory for the **HalluciGuard-OKF** multi-signal hallucination detection pipeline. It implements:
- Factual verification across 3-way decision classes (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`)
- OKF structured evidence retrieval with provenance and temporal validity
- Baseline comparison, neural multi-signal training, ablation studies, and error analysis.

---
"""))

# 01. Environment setup
cells.append(nbf.v4.new_markdown_cell("""## 01. Environment Setup
Verify Python runtime, device availability (CPU/CUDA), and directory structures.
"""))
cells.append(nbf.v4.new_code_cell("""import sys
import os
import random
from pathlib import Path
import torch
import numpy as np

# Set working directory to project root if running from notebooks/
if Path.cwd().name == "notebooks":
    os.chdir("..")

print(f"Working Directory: {os.getcwd()}")
print(f"Python Executable: {sys.executable}")
print(f"PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
"""))

# 02. Import dependencies
cells.append(nbf.v4.new_markdown_cell("""## 02. Import Dependencies
Import core ML libraries, transformers, evaluation metrics, and HalluciGuard-OKF modules.
"""))
cells.append(nbf.v4.new_code_cell("""import json
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from okf.schemas.okf_spec import OKFArtifact
from okf.validator.validator import OKFValidator
from knowledge.store import OKFKnowledgeStore
from ml.claim_extraction.extractor import ClaimExtractor
from retrieval.hybrid import HybridRetriever
from ml.nli.nli_engine import NLIEngine
from verification.semantic import SemanticSimilarityEngine
from verification.provenance import ProvenanceAnalyzer
from verification.freshness import FreshnessAnalyzer
from ml.scoring.features import SignalFeatureExtractor
from ml.scoring.models import MultiSignalScorer, HeuristicThresholdScorer, ProductionScorer
from datasets.loader import BenchmarkDataLoader, LABEL_MAP, INV_LABEL_MAP
from evaluation.metrics import evaluate_predictions, compute_calibration_error
"""))

# 03. Dataset loading
cells.append(nbf.v4.new_markdown_cell("""## 03. Dataset Loading
Load the curated multi-domain benchmark dataset conforming to the Section 18 schema.
"""))
cells.append(nbf.v4.new_code_cell("""benchmark_path = "datasets/benchmark/benchmark_data.json"
with open(benchmark_path, "r", encoding="utf-8") as f:
    raw_benchmark = json.load(f)

print(f"Loaded {len(raw_benchmark)} benchmark scenarios containing {sum(len(ex['claims']) for ex in raw_benchmark)} claims.")
"""))

# 04. Dataset inspection
cells.append(nbf.v4.new_markdown_cell("""## 04. Dataset Inspection
Examine benchmark samples, schema fields, and annotator agreement scores.
"""))
cells.append(nbf.v4.new_code_cell("""sample_ex = raw_benchmark[0]
print("Scenario ID:", sample_ex["example_id"])
print("Query:", sample_ex["query"])
print("LLM Response:", sample_ex["llm_response"])
print("\nFirst Claim Breakdown:")
print(json.dumps(sample_ex["claims"][0], indent=2))
"""))

# 05. Data visualization
cells.append(nbf.v4.new_markdown_cell("""## 05. Data Visualization
Visualize distribution of domains and claim categories across the benchmark.
"""))
cells.append(nbf.v4.new_code_cell("""domains = [ex["domain"] for ex in raw_benchmark]
domain_counts = {d: domains.count(d) for d in set(domains)}

plt.figure(figsize=(8, 4))
plt.bar(list(domain_counts.keys()), list(domain_counts.values()), color="#3b82f6")
plt.title("Benchmark Domain Distribution")
plt.xlabel("Domain")
plt.ylabel("Scenario Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
"""))

# 06. Class distribution analysis
cells.append(nbf.v4.new_markdown_cell("""## 06. Class Distribution Analysis
Analyze representation of `SUPPORTED`, `CONTRADICTED`, and `UNVERIFIED` classes.
"""))
cells.append(nbf.v4.new_code_cell("""all_claims = [c for ex in raw_benchmark for c in ex["claims"]]
labels = [c["gold_label"] for c in all_claims]
label_counts = {l: labels.count(l) for l in set(labels)}

print("Claim Label Counts:", label_counts)
plt.figure(figsize=(6, 4))
plt.pie(list(label_counts.values()), labels=list(label_counts.keys()), autopct='%1.1f%%', colors=["#10b981", "#ef4444", "#f59e0b"])
plt.title("Gold Claim Label Distribution (3-Way)")
plt.show()
"""))

# 07. Data preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 07. Data Preprocessing
Extract 10 normalized features per claim using the multi-signal feature extractor.
"""))
cells.append(nbf.v4.new_code_cell("""data_loader = BenchmarkDataLoader(benchmark_path=benchmark_path, random_seed=SEED)
preprocessed = data_loader.load_and_preprocess(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

X_train, y_train, risk_train, _ = preprocessed["splits"]["train"]
X_val, y_val, risk_val, _ = preprocessed["splits"]["val"]
X_test, y_test, risk_test, meta_test = preprocessed["splits"]["test"]

print(f"Train samples: {len(y_train)} | Val samples: {len(y_val)} | Test samples: {len(y_test)}")
print(f"Feature vector dimensionality: {X_train.shape[1]}")
"""))

# 08. Train/validation/test split
cells.append(nbf.v4.new_markdown_cell("""## 08. Train/Validation/Test Split Validation
Confirm that class stratification is maintained and verify zero data leakage.
"""))
cells.append(nbf.v4.new_code_cell("""print("Train class distribution:", np.bincount(y_train))
print("Val class distribution:  ", np.bincount(y_val))
print("Test class distribution: ", np.bincount(y_test))
print("Scaler mean parameters computed strictly from train:", preprocessed["scaler"].mean_.round(3))
"""))

# 09. Baseline model
cells.append(nbf.v4.new_markdown_cell("""## 09. Baseline Model
Evaluate the heuristic rule-based threshold baseline (Experiments A/B/C in README).
"""))
cells.append(nbf.v4.new_code_cell("""baseline = HeuristicThresholdScorer()
X_test_unscaled = preprocessed["unscaled_splits"]["test"]

base_preds = []
base_probs = []
for vec in X_test_unscaled:
    p_cls, conf, _ = baseline.predict(vec)
    base_preds.append(p_cls)
    p = [0.05, 0.05, 0.05]
    p[p_cls] = conf
    base_probs.append([v / sum(p) for v in p])

baseline_results = evaluate_predictions(y_test, np.array(base_preds), np.array(base_probs))
print(f"Baseline Test Accuracy: {baseline_results['accuracy']:.4f}")
print(f"Baseline Test Macro F1: {baseline_results['macro_f1']:.4f}")
print(f"Baseline Test ECE:      {baseline_results['expected_calibration_error']:.4f}")
"""))

# 10. Proposed model
cells.append(nbf.v4.new_markdown_cell("""## 10. Proposed Model Architecture
Instantiate the PyTorch `MultiSignalScorer` residual neural network.
"""))
cells.append(nbf.v4.new_code_cell("""model = MultiSignalScorer(input_dim=15, hidden_dims=(64, 32), dropout=0.20, num_classes=3)
print(model)
"""))

# 11. Model architecture visualization
cells.append(nbf.v4.new_markdown_cell("""## 11. Model Architecture Visualization
Inspect trainable parameters, layer dimensions, and residual skip connections.
"""))
cells.append(nbf.v4.new_code_cell("""total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {total_params}")
for name, param in model.named_parameters():
    print(f"  {name:25s} | shape: {str(list(param.shape)):15s} | grad: {param.requires_grad}")
"""))

# 12. Training
cells.append(nbf.v4.new_markdown_cell("""## 12. Training
Train model using class-weighted CrossEntropy + MSE risk loss and Cosine Annealing.
"""))
cells.append(nbf.v4.new_code_cell("""dataloaders = data_loader.create_dataloaders(preprocessed, batch_size=16)

cls_criterion = torch.nn.CrossEntropyLoss(weight=preprocessed["class_weights"])
reg_criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

epochs = 50
best_val_f1 = -1.0
train_losses = []
val_f1s = []

for epoch in range(1, epochs + 1):
    model.train()
    ep_loss = 0.0
    for feats, lbls, rsk in dataloaders["train"]:
        optimizer.zero_grad()
        logits, pred_risk = model(feats)
        loss = cls_criterion(logits, lbls) + 0.5 * reg_criterion(pred_risk, rsk)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    val_preds, val_lbls_list = [], []
    with torch.no_grad():
        for feats, lbls, _ in dataloaders["val"]:
            logits, _ = model(feats)
            val_preds.extend(torch.argmax(logits, dim=-1).numpy().tolist())
            val_lbls_list.extend(lbls.numpy().tolist())

    p, r, f1, _ = precision_recall_fscore_support(val_lbls_list, val_preds, average="macro", zero_division=0)
    train_losses.append(ep_loss / len(dataloaders["train"]))
    val_f1s.append(f1)

    if f1 > best_val_f1:
        best_val_f1 = f1
        torch.save(model.state_dict(), "models/multisignal_head/best_model.pt")

print(f"Training completed. Best Validation Macro F1: {best_val_f1:.4f}")
"""))

# 13. Validation
cells.append(nbf.v4.new_markdown_cell("""## 13. Validation Analysis
Plot training loss and validation Macro-F1 progression.
"""))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, color="#ef4444", label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True, alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(val_f1s, color="#10b981", label="Val Macro F1")
plt.title("Validation Macro F1")
plt.xlabel("Epoch")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()
"""))

# 14. Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 14. Evaluation on Held-Out Test Set
Evaluate the best checkpoint on the isolated test partition.
"""))
cells.append(nbf.v4.new_code_cell("""model.load_state_dict(torch.load("models/multisignal_head/best_model.pt"))
model.eval()

test_preds, test_probs = [], []
with torch.no_grad():
    for feats, _, _ in dataloaders["test"]:
        logits, _ = model(feats)
        probs = torch.softmax(logits, dim=-1)
        test_preds.extend(torch.argmax(probs, dim=-1).numpy().tolist())
        test_probs.extend(probs.numpy().tolist())

proposed_results = evaluate_predictions(y_test, np.array(test_preds), np.array(test_probs))
print(json.dumps(proposed_results, indent=2))
"""))

# 15. Confusion matrix
cells.append(nbf.v4.new_markdown_cell("""## 15. Confusion Matrix
Plot confusion matrix across the 3 decision classes.
"""))
cells.append(nbf.v4.new_code_cell("""cm = np.array(proposed_results["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 4))
cax = ax.matshow(cm, cmap="Blues")
plt.title("Proposed Model Confusion Matrix", pad=20)
fig.colorbar(cax)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["SUP", "CON", "UNV"])
ax.set_yticklabels(["SUP", "CON", "UNV"])
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='red' if cm[i, j] > 0 else 'black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
"""))

# 16. Calibration & ECE
cells.append(nbf.v4.new_markdown_cell("""## 16. Probability Calibration (ECE)
Measure calibration error between confidence and empirical accuracy.
"""))
cells.append(nbf.v4.new_code_cell("""ece = proposed_results["expected_calibration_error"]
print(f"Expected Calibration Error (ECE): {ece:.4f}")
print("Temperature parameter learned value:", float(model.temperature.data.item()))
"""))

# 17. Error analysis
cells.append(nbf.v4.new_markdown_cell("""## 17. Error Analysis
Analyze specific claims misclassified by the model and determine root causes.
"""))
cells.append(nbf.v4.new_code_cell("""for idx in range(len(y_test)):
    if y_test[idx] != test_preds[idx]:
        meta = meta_test[idx]
        print(f"Claim: '{meta.get('claim_text')}'")
        print(f"  True Label:      {INV_LABEL_MAP[y_test[idx]]}")
        print(f"  Predicted Label: {INV_LABEL_MAP[test_preds[idx]]}")
        print(f"  Pattern:         {meta.get('hallucination_pattern')}")
        print("-" * 50)
"""))

# 18. Comparison with baseline
cells.append(nbf.v4.new_markdown_cell("""## 18. Comparison with Baseline
Direct benchmark comparison between Heuristic Baseline and Proposed MultiSignalScorer.
"""))
cells.append(nbf.v4.new_code_cell("""print(f"{'Metric':<25} | {'Baseline':<12} | {'Proposed':<12} | {'Improvement':<12}")
print("-" * 65)
for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "expected_calibration_error"]:
    b_val = baseline_results[m]
    p_val = proposed_results[m]
    diff = p_val - b_val
    sign = "+" if diff >= 0 else ""
    print(f"{m:<25} | {b_val:<12.4f} | {p_val:<12.4f} | {sign}{diff:<12.4f}")
"""))

# 19. Ablation experiments
cells.append(nbf.v4.new_markdown_cell("""## 19. Ablation Experiments
Measure drop in performance when removing individual verification signals.
"""))
cells.append(nbf.v4.new_code_cell("""ablation_configs = {
    "Full Model": None,
    "No NLI Signals": [1, 2, 3, 13, 14],
    "No Freshness Signal": [7],
    "No Provenance Signal": [5, 6],
    "No Semantic Similarity": [0],
}

print(f"{'Ablation Configuration':<25} | {'Macro F1':<12} | {'F1 Drop':<12}")
print("-" * 55)
base_f1 = proposed_results["macro_f1"]

for name, zero_indices in ablation_configs.items():
    X_abl = X_test.copy()
    if zero_indices:
        X_abl[:, zero_indices] = 0.0
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(X_abl))
        p_abl = torch.argmax(logits, dim=-1).numpy()
    _, _, f1_abl, _ = precision_recall_fscore_support(y_test, p_abl, average="macro", zero_division=0)
    drop = base_f1 - f1_abl
    print(f"{name:<25} | {f1_abl:<12.4f} | -{drop:<12.4f}")
"""))

# 20. Save final model
cells.append(nbf.v4.new_markdown_cell("""## 20. Save Final Model Artifacts
Export model state dict, feature scaler, and metadata schema.
"""))
cells.append(nbf.v4.new_code_cell("""import json
out_dir = Path("models/multisignal_head")
out_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), out_dir / "best_model.pt")
scaler = preprocessed["scaler"]
with open(out_dir / "feature_scaler.json", "w", encoding="utf-8") as f:
    json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f, indent=2)
print("Saved best_model.pt and feature_scaler.json to", out_dir)
"""))

# 21. Test inference
cells.append(nbf.v4.new_markdown_cell("""## 21. Test End-to-End Inference
Execute inference on a test scenario using the complete `HalluciGuardPipeline`.
"""))
cells.append(nbf.v4.new_code_cell("""from inference import HalluciGuardPipeline

pipeline = HalluciGuardPipeline()
result = pipeline.analyze(
    response_text="Tesla was founded in 2008 by Elon Musk.",
    query="Who founded Tesla?",
    domain="business"
)

print(json.dumps(result, indent=2))
"""))

nb["cells"] = cells

notebook_path = Path("notebooks/08_model_training.ipynb")
notebook_path.parent.mkdir(parents=True, exist_ok=True)
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated notebook: {notebook_path}")
