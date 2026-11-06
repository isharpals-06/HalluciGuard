# HalluciGuard Project Context Report

> **Implementation update (2026-11-05):** This report is the pre-development audit snapshot. Since it was written, primary training features were changed to use retrieved corpus evidence (gold evidence is available only in explicit oracle mode), splits are grouped by scenario, runtime modes are explicit, top-K evidence and conflict conditions are represented, validation-only temperature/abstention selection and additional metrics were added, and local external-dataset adapters were added. Current implementation details and results are in [documentation/IMPLEMENTATION_STATUS.md](documentation/IMPLEMENTATION_STATUS.md), [documentation/RESEARCH_REPORT.md](documentation/RESEARCH_REPORT.md), and [evaluation/results/current_training_run.json](evaluation/results/current_training_run.json). Existing RAGTruth/TRIVIA+ data was not downloaded; no external benchmark run or perturbation experiment is claimed.

**Audit scope:** Read-only repository inspection on 2026-11-05. This document distinguishes source-code evidence, checked-in artifacts, and claims made only in project documentation. No training, dataset download, or fresh test run was performed.

## Table of Contents

1. [Executive Project Summary](#section-1--executive-project-summary)
2. [Project Identity and Scope](#section-2--project-identity-and-scope)
3. [Repository Structure](#section-3--repository-structure)
4. [Existing Architecture](#section-4--existing-architecture)
5. [Detailed Implementation Audit](#section-5--detailed-implementation-audit)
6. [Dataset and Data Pipeline Audit](#section-6--dataset-and-data-pipeline-audit)
7. [Existing Detection Methodology](#section-7--existing-detection-methodology)
8. [Model and Training Audit](#section-8--model-and-training-audit)
9. [Experimental Evaluation Audit](#section-9--experimental-evaluation-audit)
10. [Testing and Reliability Audit](#section-10--testing-and-reliability-audit)
11. [Existing User Interface and Deployment](#section-11--existing-user-interface-and-deployment)
12. [Environment and Computational Feasibility](#section-12--environment-and-computational-feasibility)
13. [Security and Engineering Quality](#section-13--security-and-engineering-quality)
14. [Gap Analysis Against the Final Research Vision](#section-14--gap-analysis-against-the-final-research-vision)
15. [Research Risks and Methodological Weaknesses](#section-15--research-risks-and-methodological-weaknesses)
16. [Recommended Final Architecture](#section-16--recommended-final-architecture)
17. [Prioritized Development Roadmap](#section-17--prioritized-development-roadmap)
18. [Critical Blockers and Unanswered Questions](#section-18--critical-blockers-and-unanswered-questions)
19. [Final Project Readiness Assessment](#section-19--final-project-readiness-assessment)
20. [Handover Context for ChatGPT](#section-20--handover-context-for-chatgpt)

## SECTION 1 — Executive Project Summary

HalluciGuard-OKF is a Python research prototype for claim-level factual verification against a local collection of structured OKF knowledge artifacts. The main implemented route accepts response text, splits it into claims, retrieves local evidence using hybrid lexical/dense retrieval, computes similarity, NLI-like, provenance, and freshness features, passes the feature vector to a three-class MLP/scoring wrapper, and returns claim and response summaries. There are CLI and FastAPI routes. Core orchestration is in `inference.py:34-91,123-237,273-357`; CLI entry points are in `cli.py`; web routes are in `web/app.py`.

**Overall status: PARTIALLY IMPLEMENTED as a research prototype; UNVERIFIED in this audit by execution.** Source code for a functioning local pipeline, a small custom dataset, a saved head, metric artifacts, tests, and documentation exists. However, core signals are partly heuristic, evidence aggregation/conflict handling is absent, benchmark features leak gold-provided evidence into inputs, and no retrieval perturbation or external benchmark framework was identified. “Calibrated,” “epistemic,” “verified,” and “production-ready” labels in docs overstate what code and artifacts establish.

Major strengths: modular Python structure; explicit SUPPORTED / CONTRADICTED / UNVERIFIED taxonomy; absence-of-evidence handling; local CPU-oriented inference; Pydantic knowledge schema; benchmark/experiment code; CLI/web surfaces. Major gaps: no explicit conflicting-evidence class/flag, no independently established calibration, no controlled perturbation experiments, narrow custom corpus, tiny test split, and unreconciled documentation/artifact inconsistencies. Research readiness for the target retrieval-robustness study is low; the code is a reusable prototype foundation but distant from the stated final objective.

## SECTION 2 — Project Identity and Scope

- **Name:** HalluciGuard / HalluciGuard-OKF (package name `halluciguard-okf`, `pyproject.toml:5-10`).
- **Current documented objective:** explainable claim-level verification using OKF structured evidence, provenance/freshness signals, multi-signal classification, and abstention; see `documentation/RESEARCH_PROBLEM.md:181-196`.
- **Observed functional scope:** local knowledge ingestion, claim heuristics, retrieval, rule-based NLI fallback, scoring, response aggregation, local CLI and FastAPI interface.
- **Intended users:** researchers and users evaluating generated text against a curated local knowledge base; docs also claim interactive demonstration use.
- **Current non-functional scope:** CPU device is hardcoded in training and scoring paths (`train.py:110-112`; `ml/scoring/models.py:169-175`). No deployment/container configuration was found.
- **Target objective (proposed, not achieved):** robust and calibrated claim-level hallucination detection under controlled retrieval perturbations, with supported, contradicted, insufficient-evidence labels; conflict treated as an evidence condition; selective abstention; evaluation on TRIVIA+, RAGTruth and optionally RAGuard.
- **Scope mismatch:** existing project centers on OKF/provenance and a small bespoke knowledge base. Target centers on robustness to retrieval changes and broad external benchmarks. Existing code does not implement the target’s perturbation protocol or establish benchmark compatibility.

## SECTION 3 — Repository Structure

Relevant tree (caches and virtual environment omitted):

```text
.
├── cli.py, inference.py, train.py
├── configs/                 model, training, scoring, OKF schema configuration
├── datasets/                loader.py; benchmark/benchmark_data.json
├── evaluation/              metrics.py, experiment.py, results/, plots/
├── knowledge/               store.py; artifacts/curated_knowledge.json
├── ml/                      claim_extraction/, nli/, providers/, scoring/
├── models/multisignal_head/ saved model weights, scaler files, metadata
├── okf/                     schemas/, generator/, validator/
├── retrieval/               sparse.py, dense.py, hybrid.py
├── verification/            semantic.py, provenance.py, freshness.py
├── web/                     app.py; templates/index.html
├── tests/                   unit/, integration/, adversarial/
├── notebooks/               08_model_training.ipynb
├── scripts/                 README/notebook generation and benchmark expansion scripts
├── documentation/           setup, architecture, methodology, audits, reports
├── pyproject.toml, requirements.txt, .env.example
└── README.md
```

Key responsibilities: `inference.py` orchestrates runtime; `datasets/loader.py` turns bespoke JSON into signal vectors and splits; `train.py` trains the MLP; `evaluation/experiment.py` evaluates models and writes metrics/plots; `retrieval/` implements retrieval; `knowledge/store.py` loads validated OKF records; `ml/` contains extraction, NLI, provider, and scoring components; `verification/` contains similarity/provenance/freshness; `okf/` defines and validates the artifact format. `cli.py` and `web/app.py` expose local verification and evaluation. Tests cover representative unit, integration, and adversarial paths.

There are extensive overlapping audit and status documents under `documentation/`, including `documentation/audit/`, `documentation/final_audit/`, `documentation/INITIAL_CODEBASE_AUDIT.md`, `documentation/IMPLEMENTATION_STATUS.md`, `documentation/FINAL_COMPLETION_REPORT.md`, and `documentation/RESEARCH_REPORT.md`. These are claims to be cross-checked, not authoritative proof. `notebooks/08_model_training.ipynb` exists; its execution history was not established in this audit. Generated caches and `.venv` exist. Model artifacts include `.pt`, `.safetensors`, JSON and pickle scaler variants.

**Repository/Git:** `git status` failed with “not a git repository”; no Git history, branch, or dirty/clean status can be asserted.

## SECTION 4 — Existing Architecture

| Stage | Current implementation and contract | Status / limits |
|---|---|---|
| Input | `HalluciGuardPipeline.analyze(response_text, query, domain, language, abstain_threshold)`; rejects blank response (`inference.py:93-121`). | IMPLEMENTED; no explicit comprehensive schema/length validation at API boundary found here. |
| Claim extraction | `ClaimExtractor.extract_claims` sentence split and regex clause decomposition; returns `ExtractedClaim` objects (`ml/claim_extraction/extractor.py:12-83`). | IMPLEMENTED; deterministic patterns, not semantic/LLM atomicity validation. Input truncates at 4096 chars. |
| Evidence store | `OKFKnowledgeStore` loads JSON and accepts artifacts passing Pydantic validator (`knowledge/store.py:23-36`). | IMPLEMENTED; checked-in corpus has 31 records. |
| Retrieval | `HybridRetriever.retrieve(query, top_k, entity_filter)` combines BM25 and dense retrieval (`retrieval/hybrid.py:13-42`). Inference requests top 3 (`inference.py:146-149`). | IMPLEMENTED; runtime then uses only first result and threshold (`inference.py:151-162`), so no true multi-evidence aggregation. Dense backend may fall back to TF-IDF; see `retrieval/dense.py`. |
| Evidence signals | Similarity, NLI-like output, provenance and freshness computed for top artifact (`inference.py:163-199`). | PARTIALLY IMPLEMENTED; active NLI defaults to rules (`ml/nli/nli_engine.py:13-28,43-67`); no disagreement/coverage across multiple documents. |
| Feature construction | Ten clipped features (`ml/scoring/features.py:10-56`): similarity, entailment/contradiction/neutral, evidence found, source/provenance/freshness, retrieval and extraction confidence. | IMPLEMENTED; “coverage” is a boolean evidence flag. |
| Classification/confidence | `ProductionScorer` wraps three-class `MultiSignalScorer`, risk output, confidence/abstention (`ml/scoring/models.py:93-150,156+`). | PARTIALLY IMPLEMENTED; a trained temperature parameter is not evidence of fitted probability calibration. |
| Decision overrides | Inference forces supported/contradicted when heuristic NLI thresholds exceed .80 (`inference.py:247-257`). | IMPLEMENTED, but override sets `abstained=False` without clearing `abstention_reason` and uses one top artifact; threshold 0 default means configured abstention may be off. |
| Conflict | No explicit conflict state surfaced; three class taxonomy maps neutral/no-evidence to UNVERIFIED. | MISSING as a first-class evidence condition. |
| Output | Per-claim status, scores, single evidence payload, explanation and response aggregates (`inference.py:273-357`). | IMPLEMENTED; “overall” rules label any contradiction HALLUCINATED, all unverified UNVERIFIED, all supported VERIFIED. |

Configuration is split across YAML and hard-coded runtime defaults. `configs/model_config.yaml` names `roberta-large-mnli`, but no configuration-loading path was seen in `inference.py` initialization; `NLIEngine()` defaults to `use_transformer=False`. Similarity and retrieval settings are hardcoded in inference. Training reads `configs/train.yaml` (`train.py:106-128`).

## SECTION 5 — Detailed Implementation Audit

- **Claim extraction — IMPLEMENTED, not semantically validated.** Pydantic output has type/entity/temporal fields and default extraction confidence 0.95 (`ml/claim_extraction/extractor.py:12-19,41-83`). Confidence is constant by default, not learned or empirically calibrated. Regex splitting can break semantics, fail coreference, or generate fragments.
- **OKF model — IMPLEMENTED.** Typed schema, generator and validator modules exist. Knowledge store filters invalid JSON records and prints warnings (`knowledge/store.py:23-36`). “Cryptographic identity” claims in docs require separate validation; no cryptographic verification was established from this inspection.
- **Retrieval — PARTIALLY IMPLEMENTED.** BM25 and dense module plus weighted hybrid are present. Documentation says SentenceTransformers, while final audit docs acknowledge a TF-IDF fallback when the package is unavailable. Runtime artifact selection is top-1, limiting contradictory or corroborating evidence use.
- **NLI — PARTIALLY IMPLEMENTED.** Transformer path is optional and defaults off; main path uses lexical overlap, relation regexes, temporal/numeric rules (`ml/nli/nli_engine.py:13-67,69+`). Output key names resemble probabilities but rule outputs are hand-set scores, not model-calibrated posterior probabilities.
- **Feature/scorer — PARTIALLY IMPLEMENTED.** Ten feature values are fed to an MLP and auxiliary risk head. Inference has direct rule overrides after MLP (`inference.py:232-257`). Calibration and epistemic uncertainty terminology is not independently supported by code evidence of calibration fitting or uncertainty ensembles.
- **Training — IMPLEMENTED pipeline, RESEARCH VALIDITY CONCERN.** `train.py` loads YAML, seeds RNGs, builds train/validation/test, computes baseline, trains with weighted cross-entropy + risk MSE and AdamW, scheduler and early stopping (`train.py:35-75,106-176,178+`). Training loader generates features using each example’s `evidence` field and `gold_label` (`datasets/loader.py:74-133`); evaluation thus may encode gold-evidence availability/content, unlike runtime retrieval, and splits occur after feature extraction.
- **Evaluation — PARTIALLY IMPLEMENTED.** Code computes accuracy/precision/recall/F1/ECE/confusion matrix and risk-coverage/reliability plots (`evaluation/metrics.py:16-90`; `evaluation/experiment.py:52-87`). It is unclear whether checked-in artifacts were generated by the current source revision. No experiment tracking system found.
- **Provider integrations — PARTIALLY IMPLEMENTED.** Local/cloud provider adapters exist, but core verification uses local store and signals; provider’s role in factual verification is not established. `.env.example` lists variable names only; secret values are intentionally omitted from this report.
- **Duplication/configuration:** numerous audit docs report varying completion status; `requirements.txt` says “Pinned” but entries are lower bounds (`requirements.txt:1-13`). `pyproject.toml` repeats those unpinned requirements (`pyproject.toml:14-33`). Config YAML values are not consistently the values used by inference.

## SECTION 6 — Dataset and Data Pipeline Audit

| Dataset | Observed facts | Status |
|---|---|---|
| Custom benchmark | `datasets/benchmark/benchmark_data.json`; JSON array of 72 scenario records, 84 claim records according to `documentation/DATASET.md:24-35` and checked JSON top-level count (72). Docs report 33 SUPPORTED, 29 CONTRADICTED, 22 UNVERIFIED. Each claim includes text, gold label, evidence text/source, provenance, freshness, and descriptive tags. | IMPLEMENTED custom data; source/license/annotation protocol not established. Counts are doc-reported except scenario count checked. |
| Curated OKF corpus | `knowledge/artifacts/curated_knowledge.json`; JSON top-level count 31. Docs describe 31 verified artifacts and domain categories (`documentation/DATASET.md:11-20`). | IMPLEMENTED local corpus; external source provenance and licensing require audit. |
| TRIVIA+ | No loader/data/config integration identified. | MISSING / PLANNED target only. |
| RAGTruth | No loader/data/config integration identified. | MISSING / PLANNED target only. |
| RAGuard | No loader/data/config integration identified. | MISSING / PLANNED optional target only. |

`BenchmarkDataLoader` deduplicates exact claim text before splitting and stratifies labels; feature scaler is fit on train only (`datasets/loader.py:72-80,139-162`). However, deduplication before split does not detect paraphrase or scenario leakage. Features are computed using `c["evidence"]`, which is a gold-supplied evidence field; runtime evidence comes from retrieved OKF artifacts. This is a major train/inference distribution mismatch and possible target/evidence leakage. It filters unknown labels silently (`loader.py:82-85`), assumes fields/ratios/classes adequate, and can fail stratification on small classes. No missing-value policy, official split IDs, license, or independent annotation agreement procedure was verified. Label taxonomy has no conflicting-evidence label.

## SECTION 7 — Existing Detection Methodology

The runtime is claim-level, with a three-way label map SUPPORTED=0, CONTRADICTED=1, UNVERIFIED=2 (`datasets/loader.py:22-23`; `configs/model_config.yaml:37-39`). It does not explicitly distinguish conflicting evidence from insufficient evidence. It combines lexical/dense retrieval, semantic similarity, hand-coded NLI fallback, provenance/freshness features, and a neural classification head, followed by rule overrides.

Atomic decomposition is regex/clause based. BM25 and dense retrieval exist; semantic similarity is a feature. Entailment/contradiction/neutral signals are produced, but the default engine is rule-based, and an optional transformer must be explicitly enabled. Evidence relevance is represented by retrieval score. No explicit disagreement score or multi-document evidence coverage exists. No LLM-as-a-Judge detection route was identified. Missing evidence is deliberately mapped to neutral/UNVERIFIED in code (`inference.py:200-212`, `ml/scoring/models.py:74-87`), which is sound taxonomy logic. Evidence disagreement is not operationalized.

## SECTION 8 — Model and Training Audit

- Scoring model: PyTorch MLP input 10, hidden layers 64 and 32, dropout .2, 3-class logits, auxiliary sigmoid risk output, trainable temperature (`ml/scoring/models.py:93-150`; config `configs/model_config.yaml:24-39`).
- NLI config lists `roberta-large-mnli`, max length 256, temperature 1 (`configs/model_config.yaml:18-22`), but runtime calls default `NLIEngine()` and thus uses rules (`inference.py:67-69`; `ml/nli/nli_engine.py:13-28`).
- Dense config lists `sentence-transformers/all-MiniLM-L6-v2` (`configs/model_config.yaml:6-11`), but dependency manifest does not list sentence-transformers. Audit docs describe TF-IDF fallback. No guarantee of transformer embeddings at runtime.
- Training configuration: seed 42, nominal 70/15/15 split, 80 epochs, batch 16, LR .003, AdamW, cosine schedule, patience 12 (`configs/train.yaml:1-37`). Training hardcodes CPU (`train.py:110-112`). Loss combines weighted classification and 0.5 risk regression (`train.py:55-69`).
- Checked-in artifacts: `models/multisignal_head/best_model.pt`, `.safetensors`, JSON scaler, pickle scaler and metadata. Artifact presence does not prove which artifact is loaded or its reproducibility. Metadata claims 10 input features, seed 42, test accuracy 1.0 on 13 rows; these are recorded results, not rerun.
- Model sizes/checkpoint provenance, tokenizer use, GPU memory, RAM, and training run logs are UNKNOWN. No fine-tuning of transformer was established. Reproducibility is PARTIAL: seeds/config exist, but dependencies are not pinned exactly and data/feature/test artifacts have inconsistencies.

## SECTION 9 — Experimental Evaluation Audit

Checked-in `evaluation/results/experiment_results.json` contains result records; `documentation/RESEARCH_REPORT.md:18-31` repeats some numbers. These are stored outputs and were not independently regenerated in this audit. Results include heuristic baseline 0.9231 accuracy / 0.9153 macro-F1; forced MLP test split 1.0 accuracy on support 13 (5/5/3 classes); full-benchmark forced 0.9643 accuracy / 0.9675 macro-F1 on 84; selective record reports 0.9649 accuracy/F1 at 0.6786 coverage, with 57 accepted and 27 abstained. Selective output has zero UNVERIFIED support among accepted predictions, so its macro metric is over a changed class distribution and is not directly comparable to ordinary three-class benchmark macro-F1. Reported ECE for selective output is 0.3832, higher than forced full benchmark ECE 0.3475; this does not support a claim that calibration improved. These metrics must be checked against generation code and split protocol before research claims.

`evaluation/metrics.py` implements accuracy, macro/weighted and per-class P/R/F1, confusion matrix, ECE. Risk-coverage and plots are present in `evaluation/experiment.py` and checked-in `evaluation/plots/`. Brier score is imported but not returned in `evaluate_predictions`. PR-AUC, NLL, AURC, flip rate, confidence drift, conflict F1, retrieval Recall@K, latency distributions and memory usage were not found as systematic benchmark metrics.

| Metric | Status | Evidence |
|---|---|---|
| Accuracy, P/R/F1, confusion matrix | IMPLEMENTED | `evaluation/metrics.py:44-90` |
| ECE | IMPLEMENTED | `evaluation/metrics.py:16-41` |
| Risk-coverage / selective metrics | IMPLEMENTED in experiment code; stored artifacts exist | `evaluation/experiment.py:52-87`; not freshly run |
| Brier score | PARTIAL | imported but not emitted by metric function (`evaluation/metrics.py:8-13,81-90`) |
| PR-AUC, NLL, AURC | MISSING from observed metrics implementation | — |
| Flip rate, confidence drift, conflict F1 | MISSING | No perturbation pairing/conflict label pipeline found |
| Recall@K, latency, memory | PARTIAL/MISSING | runtime reports per-request elapsed time (`inference.py:333-355`); no retrieval recall or resource benchmark |

No external-benchmark or retrieval-perturbation experiments were found. Ablation claims exist in docs, but execution pathway and reproducible configurations require verification against `evaluation/experiment.py` before treating all listed ablations as validated.

## SECTION 10 — Testing and Reliability Audit

Test files include claim extraction, end-to-end pipeline, OKF validation, retrieval/verification, CLI/web integration and adversarial security tests (`tests/unit/`, `tests/integration/`, `tests/adversarial/`). A checked-in report claims 29 passed, 0 failed, 0 skipped in 14.28 seconds on Python 3.11.15 / pytest 9.2.0 (`documentation/final_audit/TEST_EXECUTION_REPORT.md:1-63`). This audit did **not** execute tests; those results are DOCUMENTED, not VERIFIED here.

Tests cover basic supported, contradicted, unverified examples, schema checks, BM25, selected NLI rules, provenance/freshness, CLI/web and selected adversarial cases. Missing tests include leakage checks, calibration correctness, perturbation invariance, paired dataset transformations, multi-evidence conflict, external benchmark loaders, duplicate/near-duplicate split checks, and model-independent inference consistency. Notebook execution was not verified.

## SECTION 11 — Existing User Interface and Deployment

CLI is exposed through `python cli.py` and package script `halluciguard = "cli:main"` (`pyproject.toml:35-36`). `cli.py` offers verification/evaluation/benchmark/info commands per parser; exact usage is documented in `documentation/CLI_GUIDE.md`. FastAPI app in `web/app.py` serves UI and verification endpoints; `/api/verify` calls `HalluciGuardPipeline.analyze` (`web/app.py:142-152`), and development launch is `uvicorn` on port 8000 (`web/app.py:164-166`). Pipeline initializes local knowledge and models at construction (`inference.py:48-91`).

This is a local demo API/UI, not evidence of production deployment. No Docker/container, authentication, rate limiting, or deployed environment was found. Provider adapters and environment-variable names exist; no credential values are included here. Network exposure/security configuration should be resolved before non-local service.

## SECTION 12 — Environment and Computational Feasibility

`pyproject.toml` requires Python >=3.11 and lists torch, transformers, Pydantic, scikit-learn, FastAPI, PyYAML, pytest, matplotlib, safetensors, joblib and requests (`pyproject.toml:10-33`). Requirements duplicate lower bounds (`requirements.txt:1-13`), not fixed lock versions. Sentence-transformers, although named in config, is absent from manifests. Python versions 3.11 and 3.14 cache artifacts exist, but actual current interpreter was not queried. GPU/CUDA is not required by configured path; train/scorer are CPU. Runtime may download pretrained models if optional model loading is enabled, but no offline model cache status was established. Local artifacts are small (weights around 15–20 KB, data JSON tens of KB); no large checkpoints identified.

Suitability: CPU demonstration on a small local corpus appears plausible from architecture, but runtime timing and exact dependencies were not independently tested. Resource requirements beyond that are UNKNOWN. Avoid claims of real NLI/SentenceTransformer operation without establishing installed libraries, model availability, and active code path.

## SECTION 13 — Security and Engineering Quality

- Positive code evidence: bounded claim input length and empty handling (`ml/claim_extraction/extractor.py:31-50`); JSON scaler safe load class (`ml/scoring/models.py:24-42`); Pydantic validation before store insertion (`knowledge/store.py:23-36`); FastAPI blank text check (`web/app.py:142-145`).
- Serialization caveat: both `feature_scaler.pkl` and `.pt` exist. Inference may call `joblib.load` when a non-JSON scaler path is selected (`inference.py:81-91`), contradicting the blanket docs claim of “zero pickle dependencies.” Default path is JSON, but fallback path exists. `ProductionScorer` prefers safetensors when present; inspect weight-loading fallback before security assurance.
- Error handling: transformer load catches exceptions and falls back to rules (`ml/nli/nli_engine.py:30-41`); inference can silently proceed without scaler (`inference.py:81-91`). Silent fallback can obscure system mode.
- Data integrity/provenance/licensing: corpus includes source names/metadata, but source authenticity and reuse licensing were not independently audited. No credential values are exposed in this report.
- No dependency vulnerability scan or live security testing was performed. Do not claim no vulnerabilities. Hardcoded thresholds/config values and broad catch behavior affect auditability. Read-only inspection only.

## SECTION 14 — Gap Analysis Against the Final Research Vision

Status terms follow the prompt: IMPLEMENTED means code connected by source inspection; VERIFIED requires a fresh execution check, which was not performed; UNVERIFIED means implementation or artifact exists without execution/correctness proof.

| Component | Existing Status | Evidence | Required Change | Priority | Dependencies |
|---|---|---|---|---|---|
| Claim decomposition and claim-level output | IMPLEMENTED | `ml/claim_extraction/extractor.py:41-83`; `inference.py:123-149` | Improve/measure extraction quality and preserve provenance through splits | P1 | Annotated claims, evaluation |
| BM25 retrieval | IMPLEMENTED | `retrieval/hybrid.py:25-42`; `retrieval/sparse.py` | Establish reproducible retrieval metrics and corpus adapters | P1 | Corpus, labels |
| Semantic dense retrieval | PARTIALLY IMPLEMENTED | `retrieval/dense.py`; config `configs/model_config.yaml:6-11`; no package dependency | Pin and test embedder or explicitly characterize fallback | P1 | Model/package availability |
| Top-K evidence use | PARTIALLY IMPLEMENTED | `inference.py:146-162` retrieves 3, consumes top one | Aggregate multiple items and cite IDs/ranks | P1 | Retrieval |
| NLI entailment/contradiction/neutral | PARTIALLY IMPLEMENTED | `ml/nli/nli_engine.py:13-67`; default rules | Make model/fallback mode explicit, validate benchmark and calibration | P1 | NLI model, labelled pairs |
| Evidence relevance/coverage | PARTIALLY IMPLEMENTED | retrieval score and binary feature (`features.py:24-56`) | Define per-claim coverage/relevance consistently | P1 | Multi-evidence retrieval |
| Disagreement/conflict detection | MISSING | No conflict state in taxonomy/code | Add conflict condition and evaluate separately from insufficient evidence | P0 | Multiple evidence, annotations |
| Lightweight evidence aggregation | PARTIALLY IMPLEMENTED | 10-feature MLP (`ml/scoring/models.py:93-150`) but one top artifact | Train on inference-realistic, non-leaking features; compare Logistic Regression/MLP | P0 | Correct data pipeline |
| Calibration | PARTIALLY IMPLEMENTED | Temperature parameter present; ECE computed | Fit calibration on validation only; report Brier/NLL and reliability with class support | P0 | Correct held-out protocol |
| Abstention | PARTIALLY IMPLEMENTED | `ProductionScorer`; inference threshold and overrides | Select threshold on validation, guarantee overrides respect abstention, report risk/coverage | P1 | Calibration |
| TRIVIA+ / RAGTruth / RAGuard | MISSING | No integration found | Verify licenses/schema, adapter and split protocols | P1 | Dataset access/licensing |
| Retrieval perturbations | MISSING | No paired transform runner found | Implement order, distractor, position, removal, contradiction transformations with valid labels | P0 | Dataset/evidence annotations |
| Baselines | PARTIALLY IMPLEMENTED | Heuristic and MLP in `evaluation/experiment.py` | Add BM25 lexical, frozen NLI and controlled optional judge baseline | P1 | Shared test protocol |
| Final metric suite | PARTIALLY IMPLEMENTED | `evaluation/metrics.py`; risk curve | Add PR-AUC, Brier, NLL, AURC, flip/drift/conflict/retrieval/resource measures | P1 | Prediction/evidence logs |
| Reproducibility | PARTIALLY IMPLEMENTED | seed/config/artifacts; no exact dependency lock; data mismatch | Freeze data versions, splits, configs, run metadata and environment | P0 | Foundation |

## SECTION 15 — Research Risks and Methodological Weaknesses

1. **Evidence leakage and train/runtime mismatch:** loader features use benchmark gold evidence, while inference retrieves from local OKF records (`datasets/loader.py:87-120` vs `inference.py:146-199`). Performance may measure feature-label shortcuts rather than the proposed retrieval system.
2. **Tiny benchmark and test set:** 84 claims, 13 test rows per documentation. A single error moves accuracy substantially; no broad generalization claim is justified.
3. **Repeated/related claims:** exact text deduplication does not prevent scenario or semantic duplicates crossing random splits. Entity/fact reuse may inflate test score.
4. **Taxonomy mismatch:** target calls for conflicting evidence as a condition; current three labels collapse conflict into one forced class or UNVERIFIED.
5. **Evidence faithfulness vs factual correctness:** curated local evidence and labels reflect corpus coverage, not necessarily world truth. “UNVERIFIED” means corpus does not support decision, not factual falsehood.
6. **Retrieval bias and narrow knowledge base:** 31 artifacts favor known facts; false no-hit cases and source authority patterns can dominate outcomes.
7. **Calibration claims:** temperature parameter and ECE metric do not establish fitted calibration. Stored ECE 0.3832 selective vs 0.3475 forced full benchmark undermines broad calibration-improvement claims. Selective score class distribution changes.
8. **Invalid transformed labels:** removing support or inserting contradiction may change the correct evidence condition; each perturbation needs a principled label policy and paired original/transformed records.
9. **Inadequate baseline fairness:** heuristic/MLP comparisons may differ in evidence input and split; LLM judge and frozen NLI baselines not established.
10. **No statistical uncertainty / multiple seeds:** results report point estimates; no confidence intervals or repeated-seed variation were confirmed in source outputs.
11. **Novelty and documentation overclaim:** “complete/verified” and research claims should be narrowed until source execution and benchmark study support them.

## SECTION 16 — Recommended Final Architecture

**Target architecture, not current state:**

1. Ingest dataset examples with immutable example IDs, answer/claim annotations, source evidence, dataset provenance/license, and official splits. Validate schema and detect duplicate/near-duplicate entities across partitions.
2. Extract or consume human-annotated atomic claims; preserve extraction confidence and evaluate extraction separately.
3. Apply a retrieval interface with BM25 and frozen embedding retriever. Save candidate IDs, scores, rank, ordering, retrieval mode and corpus snapshot.
4. Run a frozen NLI model on each claim/evidence pair to produce entailment, contradiction and neutrality; identify disagreement among multiple evidence items and separate “conflict” condition from the three claim outcomes.
5. Aggregate evidence-level features (max/mean entailment, contradiction, neutrality, relevance, coverage, disagreement, provenance) with transparent baseline models and a small classifier.
6. Calibrate probabilities using validation-only temperature scaling or isotonic calibration; choose abstention thresholds on validation and freeze them before test. Define conflict and insufficient-evidence routes explicitly.
7. Run paired perturbation transforms with lineage to source example: permutation, irrelevant distractors, position, support removal, contradiction insertion. Record transformed label policy and reject invalid transformations.
8. Compare lexical, frozen-NLI, optional LLM judge and evidence-aggregation baselines under same examples, evidence budget, split, and metrics. Report macro/per-class metrics, PR-AUC, ECE, Brier, NLL, coverage-risk/AURC, flip/drift, conflict F1, retrieval recall, latency and memory, confidence intervals and ablations.

**Retain:** OKF schema/store where useful, sparse retrieval, current feature/scoring abstractions, tests, CLI/API as demonstration surfaces, and artifact explanations. **Modify:** loader and feature source, retriever to preserve/use top-K, NLI initialization/config, abstention/calibration and experiment protocol. **Add:** benchmark adapters, perturbation harness, conflict analysis, robust metrics, run manifests and split/leakage checks. **Remove or de-emphasize:** any undocumented production/calibration certainty and pseudo-probability semantics; avoid wholesale rewrite unless component tests show a need.

## SECTION 17 — Prioritized Development Roadmap

| Phase | Objectives / reusable code | Required changes and dependencies | Deliverables / acceptance criteria | Risks |
|---|---|---|---|---|
| 0 Foundation | Reuse configs, loader, tests, metadata | Pin compatible environment; define immutable dataset and splits; inspect licenses; fix evidence-feature mismatch; record run manifest. | Clean setup; data validation; reproducible split IDs; leakage audit passes; documented CPU baseline. | Benchmark provenance/license unknown. |
| 1 Claim verification | Retain extractor, retrieval, OKF, feature/scorer APIs | Validate extraction; ensure runtime retrieval is actual feature source; consume top-K; enable a known frozen NLI mode; add conflict condition. | Claim/evidence records have traceable IDs; each class and insufficient/conflict case has tests and annotation policy. | Rule and transformer outputs may differ. |
| 2 Baselines | Retain heuristic and experiment scaffold | Official dataset adapters; shared splits and evidence budgets; BM25, frozen NLI, aggregation baseline. | Comparable per-class metrics for all methods on frozen test sets. | Taxonomy compatibility and licensing. |
| 3 Perturbations | Reuse retrieval and experiment outputs | Paired ordering/distractor/position/removal/contradiction generators; label validity checks. | Deterministic transformation records; flip rate and confidence drift; label adjudication rules. | Some transforms invalidate original labels. |
| 4 Calibration/abstention | Reuse scorer wrapper, ECE and risk curve | Validation-only temperature/isotonic fit, threshold selection; ensure rule overrides respect abstention. | Calibration plots and Brier/NLL/ECE; risk-coverage and AURC; accepted coverage reported by class. | Small validation set leads to unstable thresholds. |
| 5 Research experiments | Reuse metrics, plots, ablation framework | Cross-dataset evaluation, ablations, multi-seed/statistical uncertainty, failure analysis. | Paired results, confidence intervals, per-condition analysis and reproducible output bundle. | Computational budget and cross-domain shifts. |
| 6 Final deliverables | Reuse docs/API as appropriate | Write research report, reproducibility guide, model/data cards and final source validation. | Independent rerun reproduces tables; claims match measured scope; final tests pass. | Publication claims may be constrained by data rights and sample size. |

Dependencies are sequential: phase 0 data validity precedes model comparison; phase 1 evidence signals precede perturbations; calibration depends on stable validation protocol; final claims depend on full experiment reruns.

## SECTION 18 — Critical Blockers and Unanswered Questions

- **Architecture (critical):** Is HalluciGuard intended to study OKF specifically or general retrieval perturbations? Default: retain OKF as one evidence representation, make retrieval interface corpus-agnostic.
- **Methodology (critical):** What exactly constitutes conflicting evidence, insufficient evidence, and hallucination? Default: annotate evidence condition independently from claim truth/faithfulness; never equate no retrieval with falsehood.
- **Data (critical):** What is the source, license, annotation protocol and intended split of the custom benchmark? Default: document provenance/rights before expanding or publishing; acquire external datasets only under their terms.
- **Leakage (critical):** Why does training use gold evidence while runtime retrieves its own evidence? Default: train/evaluate on the same retrieval-generated feature path or separately label oracle-evidence experiments.
- **Evaluation:** Which numbers in `experiment_results.json` correspond to which code/config revision and sample split? Why is selective score’s accepted set missing UNVERIFIED support? Default: regenerate after protocol cleanup and report class coverage.
- **Calibration:** Was temperature actually optimized, and on which split? Default: fit on validation only; reserve test for one final report.
- **Compute:** Is optional transformer NLI intended to run offline? Is model download allowed, and what memory budget applies? UNKNOWN. Default: CPU-capable frozen model with explicit cached model ID and fallback reporting.
- **Engineering:** Reconcile config values with hardcoded runtime defaults; remove ambiguity around pickle scaler fallback; exact dependency lock missing.
- **Git/history:** No `.git` metadata available in the project root, so source provenance/change state is unresolved.
- **Optional:** Include LLM-as-a-Judge only if cost, reproducibility and prompt/version controls are feasible; do not make it a required baseline by default.

## SECTION 19 — Final Project Readiness Assessment

Scores are audit judgments based on source/artifact inspection, not quality guarantees. No tests were run during this audit.

| Area | Rating | Justification |
|---|---:|---|
| Software implementation readiness | 55/100 | End-to-end modules, CLI/API and artifacts exist; configuration drift, heuristic signals and untested current execution remain. |
| Research methodology readiness | 25/100 | Target perturbation questions not implemented; leakage and taxonomy issues undermine current evidence. |
| Dataset readiness | 20/100 | Small custom corpus exists; provenance, license, annotation procedure and official external benchmarks unresolved. |
| Experimental readiness | 35/100 | Metrics/results/plots present, but tiny sample, inconsistent selective scores and no external/perturbation study. |
| Reproducibility readiness | 40/100 | Seeds/config/history/artifacts exist; dependencies unpinned, no Git repo, data/run provenance gaps. |
| Documentation readiness | 60/100 | Broad documentation coverage, but contradictory maturity and verification claims reduce reliability. |
| Final research objective readiness | 20/100 | Core direction is represented, but robustness, conflict, robust calibration, external benchmarks and paired perturbations remain. |

**Maturity:** functional-looking local research prototype, not a validated robust research system. **Strengths:** useful modular foundations, three-way no-evidence distinction, small local artifacts, CLI/API, experimental scaffolding. **Weaknesses:** evidence leakage risk, rule-heavy inference, top-1 evidence use, narrow dataset, missing perturbation/conflict framework, unsupported calibration claims.

Five actions: (1) resolve dataset provenance and leak-free split/feature protocol; (2) align training features with actual runtime retrieval; (3) define and implement conflict-aware evidence condition and multi-evidence aggregation; (4) build paired retrieval perturbation benchmark and external adapters; (5) regenerate calibrated, statistically qualified results with complete metrics.

The existing implementation is a **usable but limited foundation** for the target objective; it requires methodological correction and substantial research infrastructure before target claims are defensible.

## SECTION 20 — Handover Context for ChatGPT

### A. Confirmed Existing Capabilities

Source-connected: deterministic claim extractor; validated local OKF JSON store; BM25 and dense/hybrid retrieval code; rule-based NLI fallback; ten-feature scorer and three-way label head; provenance/freshness signals; inference response with evidence and explanation; CLI/FastAPI code; benchmark loader; metric and experiment modules. These are IMPLEMENTED by source inspection, not freshly executed here.

### B. Existing Architecture

`cli.py` / `web/app.py` → `HalluciGuardPipeline` (`inference.py`) → claim extraction → local OKF store and hybrid retriever → top-1 evidence → similarity/NLI/provenance/freshness → ten features and scaler → MLP/rule override/abstention → per-claim and aggregate output. Separate training flow: JSON benchmark → gold evidence-derived features → random stratified split/scaler → MLP training → metrics/artifacts. That training path is not equivalent to runtime retrieval.

### C. Existing Results

Only stored results were found: 84-claim custom benchmark and reported 13-example test split; artifacts report 1.0 test accuracy, 0.9643 full-benchmark forced accuracy, 0.9675 full-benchmark macro-F1, and selective 0.9649 at 67.86% coverage. These results are not independently reproduced. Selective ECE is recorded as 0.3832 vs 0.3475 forced full benchmark, and selective accepted predictions omit the UNVERIFIED class support. Treat all metrics as provisional artifact contents, not verified findings.

### D. Critical Missing Components

Leakage-free inference-matched data pipeline; explicit conflict evidence condition; genuine multi-evidence aggregation/disagreement; reliable configurable frozen NLI and retriever; calibration fitting/validation; controlled paired perturbations; TRIVIA+/RAGTruth/RAGuard compatibility; complete robustness metrics; provenance/license and reproducibility records.

### E. Existing Code Worth Preserving

`inference.py` orchestration/output schema; `ml/claim_extraction/extractor.py`; `okf/schemas/okf_spec.py`, `okf/validator/validator.py`, `knowledge/store.py`; `retrieval/sparse.py`, `retrieval/hybrid.py`, `retrieval/dense.py`; `ml/scoring/features.py`, `ml/scoring/models.py`; `evaluation/metrics.py`, `evaluation/experiment.py`; `tests/`; CLI/web surfaces for demonstrator use. Preserve selectively and add behavior-focused tests before relying on it.

### F. Architecture Decisions Requiring Attention

Choose OKF-specific vs corpus-agnostic evidence focus; define conflict label/condition semantics; decide model fallback policy; reconcile config and runtime hardcoding; establish test protocol and score provenance; remove ambiguous unsafe scaler fallback or document it; select official external datasets and licensing.

### G. Final Research Objective

Investigate whether evidence-grounded claim-level detection can preserve prediction quality, stability and confidence calibration under incomplete, irrelevant, contradictory, noisy, misranked or differently positioned retrieval contexts. Distinguish SUPPORTED, CONTRADICTED and INSUFFICIENT EVIDENCE; detect conflicting evidence as an explicit evidence condition; evaluate selective abstention. Study RQ1 retrieval perturbation effects, RQ2 value of NLI/relevance/conflict aggregation, RQ3 calibration under degraded retrieval, RQ4 risk-coverage abstention, and RQ5 invariance to semantics-preserving order/distractor changes. These are proposed questions/hypotheses, not established results.

### H. Implementation Constraints

Project declares Python >=3.11, CPU-oriented training/inference and modest saved head/data. Exact host runtime, GPU/RAM budget, timeline and model cache availability are UNKNOWN. Dependencies are minimum-version ranges, not a lock. Do not assume external dataset access/license or network model download. Existing virtual environment/caches exist but were not used to run tests.

### I. Required Development Priorities

1. Establish dataset rights/provenance, split identity and leakage-free feature generation.
2. Align inference and training evidence/features and fix top-K aggregation.
3. Specify conflict/insufficient evidence and calibrate selective output on validation.
4. Add paired perturbation and external benchmark framework.
5. Compare fair baselines with full metrics, uncertainty estimates, and reproducible manifests.

### J. Claims That Must Not Be Made

Do not claim verified 29/29 tests for this audit; calibrated confidence; epistemic uncertainty quality; transformer NLI or SentenceTransformer embeddings active by default; general/open-domain accuracy; robust prediction under retrieval perturbations; conflict detection; TRIVIA+/RAGTruth/RAGuard integration; production security/readiness; independent reproduction of reported results; absence of all vulnerabilities.

### K. Recommended Instructions for the Next AI System

Prepare a development prompt that starts with data and evaluation validity, treats docs and stored scores as unverified until reproduced, preserves useful modules, and explicitly requires tests for leakage, mode selection, calibration, evidence conflict, top-K aggregation and perturbation labels. Require each feature to be connected end-to-end and report verified vs planned. Do not implement during prompt-writing; do not claim a research result before rerunning the defined protocol.
