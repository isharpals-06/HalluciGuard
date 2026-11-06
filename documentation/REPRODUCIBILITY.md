# Reproducibility Guide

> Current scope update (2026-11-05): complete metric and figure reproducibility is not established. The current fixed-seed custom benchmark training run is reproducible with local rule-based NLI and TF-IDF retrieval, but calibration/threshold stability and generalization are unverified. See `IMPLEMENTATION_STATUS.md` and `RESEARCH_REPORT.md`.

## 1. Reproducibility Standard

Reproducibility is an explicit project objective. Only the current local training run records exact data/config/source fingerprints and per-example held-out predictions; the broader external benchmark and robustness experiment suite has not yet been reproduced.

---

## 2. Deterministic Seeds and Environment

- **Random Seed:** `42` (fixed across Python `random`, NumPy, and PyTorch).
- **Environment Isolation:** Zero reliance on unpinned global host packages.
- **Python Version:** 3.11.x
- **OS Platform:** Windows / Linux / macOS (tested on Windows 11).

---

## 3. Step-by-Step Reproduction Protocol

### Step 1: Environment Initialization
```bash
# Create virtual environment
uv venv --python 3.11 .venv

# Activate environment
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux / macOS

# Install pinned dependencies
uv pip install -r requirements.txt
```

### Step 2: Model Training from Scratch
To re-train the neural Multi-Signal head on the benchmark dataset:
```bash
python train.py --config configs/train.yaml
```
*Expected Output:*
- Class distribution in train split: `[23, 20, 15]`
- Early stopping triggered at epoch ~13.
- Isolated test set accuracy: `1.0000`, Macro F1: `1.0000`.
- Artifacts exported to `models/multisignal_head/best_model.safetensors` and `feature_scaler.json`.

### Step 3: Run Full Experimental Suite
To reproduce the complete benchmark matrix, ablations, and plots:
```bash
python cli.py benchmark
```
*Expected Output:*
- Baseline 1 Heuristic Accuracy: `0.9231`, Macro F1: `0.9153`.
- Baseline 2 Neural MLP Test Accuracy: `1.0000`, Macro F1: `1.0000`.
- Baseline 3 Full Benchmark Forced Accuracy: `0.9643`, Macro F1: `0.9675`.
- Proposed Selective Prediction Accuracy: `0.9649`, Macro F1: `0.9649`, Coverage: `0.6786`.
- CSV exported to `evaluation/results/experiment_results.csv`.
- JSON exported to `evaluation/results/experiment_results.json`.
- Plots saved to `evaluation/plots/risk_coverage_curve.png` and `evaluation/plots/calibration_curve.png`.

### Step 4: Run Complete Test Suite
```bash
pytest tests
```
*Expected Output:*
- `29 passed in ~14 seconds`.
# Current reproducibility protocol (2026-11-05)

The primary training command is `.venv\Scripts\python.exe train.py --config configs/train.yaml`. It defaults to local retrieval evidence, uses scenario-grouped partitions, fits the feature scaler on training data, fits temperature scaling and selects the selective-risk threshold on validation only, and evaluates the test set once. Run outputs record the dataset, training-config, model-config, and source fingerprints plus scenario IDs. Per-example validation/test predictions and metrics are written to `evaluation/results/current_training_run.json`.

Current local modes are NLI `rule_based` and dense retrieval `tfidf`; no transformer weights or external datasets are fetched. To reproduce the current run, use the Python 3.11 environment and dependency pins in `requirements.txt`. This fixed-seed custom-data run is small and was not repeated across multiple seeds.

