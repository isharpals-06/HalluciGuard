# HalluciGuard Implementation Status

**Updated:** 2026-11-05. This status supersedes earlier “complete and verified” prototype claims in historical audit documents. The project remains a small research prototype; external benchmark experiments and retrieval-perturbation results are not complete.

## Connected Components

| Component | Current state | Evidence / limitation |
|---|---|---|
| Claim-level inference | IMPLEMENTED | `inference.py`; legacy `status=UNVERIFIED` remains for compatibility and canonical `claim_status=INSUFFICIENT_EVIDENCE` is also returned. |
| Evidence-condition taxonomy | PARTIAL | `SUPPORT_ONLY`, `CONTRADICTION_ONLY`, `NO_RELEVANT_EVIDENCE`, `IRRELEVANT_EVIDENCE`, `CONFLICTING_EVIDENCE`; conflict abstains and asks for review. Threshold heuristics require external validation. |
| Retrieval | IMPLEMENTED | BM25 plus explicit `tfidf` fallback is the configured local mode. SentenceTransformer is opt-in and fails visibly when unavailable. |
| NLI | PARTIAL | Explicit `rule_based` and `transformer` modes; this environment uses `rule_based`. Transformer weights were not downloaded or tested. |
| Training data path | IMPLEMENTED | Primary mode retrieves from the curated corpus; `oracle_evidence` is explicit. Split by scenario ID, scaler fit on train only, 15-feature train/inference schema. |
| Calibration | PARTIAL | Validation-only scalar temperature fitting and threshold selection exist. The small 14-example validation partition is unstable; current selected threshold accepts all validation samples and does not reduce held-out risk. |
| Metrics | PARTIAL | Accuracy, per-class/macro scores, PR-AUC, ECE, multiclass Brier, NLL, risk-coverage and AURC. Paired robustness functions/transformations exist but no full perturbation experiment was run. |
| Dataset adapters | PARTIAL | Local TRIVIA+ Parquet and RAGTruth JSONL adapters preserve official splits/annotations; no dataset files were acquired or evaluated. RAGuard not implemented. |
| Baselines | PARTIAL | Heuristic, rule-based NLI-only, Logistic Regression and MLP compared on the custom grouped holdout. No frozen transformer NLI or LLM-judge run. |
| CLI/API | IMPLEMENTED | Existing surfaces retained; tests cover prior command and endpoint paths. |
| Serialization | IMPLEMENTED with legacy artifacts | Runtime accepts JSON scaler and safe tensor/state-dict weights. Pickle scaler loading is disabled; old `.pkl` file is retained but unused. |

## Current Environment and Reproduction

- Runtime: Python 3.11.15, Windows 11, CPU only; CUDA unavailable.
- `sentence_transformers`, pandas/pyarrow, and xgboost are absent. No model or dataset downloads were made.
- Dependencies are pinned to the versions available in the audit environment in `pyproject.toml` and `requirements.txt`. Cloud API requests are an optional extra.
- Baseline before changes: 29 tests passed in 35.04 seconds.
- Latest complete suite: see `documentation/TESTING.md`; test execution was repeated after implementation.
- CPU training command: `.venv\Scripts\python.exe train.py --config configs/train.yaml`.
- The command writes model/scaler metadata to `models/multisignal_head/` and traceable per-example validation/test predictions to `evaluation/results/current_training_run.json`.

## Current Custom Benchmark Run

Training uses 84 custom claims grouped by 72 scenario IDs, with a reproducible 56/14/14 train/validation/test grouping (scenario group IDs stored in model metadata). The observed run is one seed and a small dataset; it is not evidence of generalization.

- Test MLP: accuracy 0.7857; macro-F1 0.8222; ECE 0.2013; Brier 0.2814; NLL 0.4685; macro PR-AUC 1.0000; 14 test claims.
- Test Logistic Regression baseline: macro-F1 0.8857, higher than the MLP in this run.
- Validation-selected selective policy targeted risk 0.10, selected threshold 0.0 (all validation examples accepted), and yielded held-out coverage 1.0 / risk 0.2143. It did not reduce risk on the test split.
- Temperature reached the lower bound 0.5 on 14 validation examples. Calibration claims are therefore preliminary; the low held-out ECE is not proof of reliable calibration.

These are actual outputs from the current training command, saved with data/config/source fingerprints. They must be interpreted with the small split and rules-based retrieval/NLI limitations above.

## Remaining Critical Work

1. Run the paired perturbation framework on manually validated examples and report flip rate/confidence drift by perturbation.
2. Evaluate external datasets when permitted local files and optional parsing dependencies are available; preserve label taxonomies and split lineage.
3. Investigate validation threshold/calibration instability on a larger independent dataset.
4. Add a genuine frozen NLI model baseline and dense SentenceTransformer run only when weights/dependencies are available and the run is recorded explicitly.
5. Reconcile remaining historical docs and notebook claims; run multiple seeds and grouped confidence intervals before research conclusions.
