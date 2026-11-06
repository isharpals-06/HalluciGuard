# HalluciGuard Research Report — Current Experimental Status

**Updated:** 2026-11-05. Earlier performance figures in this file were produced by a prior experiment path and are superseded by the current grouped, retrieval-derived run. This report does not claim completion of the proposed robustness study.

## Objective

HalluciGuard investigates claim-level decisions against retrieved evidence. The target study concerns reliability, confidence calibration, and selective abstention under retrieval perturbations. Missing evidence is not factual contradiction; conflicting evidence is represented as a distinct evidence condition. The current implementation only partially realizes that target.

## Current Implementation and Protocol

The primary training pipeline now derives evidence signals from the same local curated knowledge store and hybrid retriever used by inference. Gold benchmark evidence is available only through the explicit `oracle_evidence` mode. The current run uses `rule_based` NLI, `tfidf` dense fallback plus BM25, top-K=5, and a 15-dimensional feature vector. Claims from each scenario are held in one partition; scaling is fit on training rows only. Details and split IDs are recorded in `models/multisignal_head/model_metadata.json`; per-example validation/test predictions are in `evaluation/results/current_training_run.json`.

The custom dataset has 84 claims from 72 scenario IDs. The run used seed 42, with grouped split sizes 56/14/14. No TRIVIA+, RAGTruth, or RAGuard data was downloaded or run. No controlled perturbation, external benchmark, multi-seed, or confidence-interval experiment was completed.

## Results From the Current Run

| Method | Test Accuracy | Macro-F1 | ECE | Brier | NLL | Macro PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic | 0.7143 | 0.6648 | 0.2784 | 0.4895 | 0.8929 | 0.8591 |
| Rule-based NLI-only | 0.7143 | 0.6746 | 0.2778 | 0.4882 | 0.8913 | 0.8406 |
| Logistic Regression | 0.8571 | 0.8857 | 0.1469 | 0.1428 | 0.2190 | 1.0000 |
| 15-feature MLP | 0.7857 | 0.8222 | 0.2013 | 0.2814 | 0.4685 | 1.0000 |

The MLP confusion matrix is `[[6,0,0],[3,3,0],[0,0,2]]`; class supports are 6 SUPPORTED, 6 CONTRADICTED and 2 UNVERIFIED. Logistic Regression outperformed the MLP on macro-F1 in this run, so the MLP is not the strongest current method.

Temperature scaling was fit on the 14-example validation partition and hit the lower bound 0.5. The validation-selected risk target of 0.10 yielded threshold 0.0, accepting every validation item. On test it accepted all 14 examples and observed risk 0.2143, so it did not reduce test risk. The small validation set makes both calibration and threshold selection unstable. The measured test ECE is not evidence of robust calibration.

## Perturbation and External Evaluation Status

`evaluation/perturbations.py` implements deterministic evidence-order permutation, irrelevant-evidence insertion, position change, support removal and contradiction insertion, while preserving lineage and flagging transformations requiring human review. Unit tests cover determinism and conflict label updates. These are utilities, not a completed perturbation benchmark: no 300–500 example suite or independent human-reviewed subset has been run. Prediction flip rate and confidence drift functions are available, but no perturbation result is reported.

Local adapters preserve TRIVIA+ sentence labels/splits and RAGTruth span offsets and source IDs. They have not been run on official data. RAGTruth span labels have not been aligned to claims. RAGuard remains optional and unimplemented.

## Limitations and Threats to Validity

- Small custom benchmark and 14-row test split; one seed, no uncertainty intervals.
- Configured NLI is rule-based. TF-IDF is not a semantic embedding model.
- The curated knowledge base contains only 31 artifacts, limiting evidence coverage and domain scope.
- High overlap or duplicated facts across related scenarios may affect generalization despite scenario-group split.
- The MLP underperforms Logistic Regression on this split; do not claim it is superior.
- Calibration operating point selected on a tiny validation sample did not reduce test risk.
- No external datasets or controlled perturbation experiment has been run.
- Historical reports contain prior scores and completion language; they should not be cited as current results.

## Reproduction

From the repository root in the pinned Python 3.11 environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe train.py --config configs/train.yaml
```

Training overwrites the model artifacts and current-run prediction JSON. It performs CPU training only and does not download models or datasets.
