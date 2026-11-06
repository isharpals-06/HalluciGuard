# HalluciGuard-OKF: Final Independent Audit Report

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Type:** Independent Technical, Architectural, Scientific, and Security Completion Audit  
**Auditor Roles:** Senior Software Auditor, AI/ML Research Evaluator, NLP Research Engineer, Security Auditor, QA Engineer  
**Audit Date:** November 2026  
**Audited Target:** HalluciGuard-OKF Repository  
**Execution Environment:** Windows 11 (Host: Python 3.14.0b3; Active Runtime: Python 3.11.15 in `.venv`)  
**Auditor Mandate:** Evidence-based completion verification; no assumed completion, no code alterations.

---

## 1. Executive Summary

This independent audit evaluated whether **HalluciGuard-OKF** fulfills its research objectives as a runnable, experimentally evaluated, uncertainty-aware factual verification system with selective abstention.

### Overall Assessment
The project has successfully transitioned from a fragmented, broken prototype (TRL 2–3) into a **functional, runnable, secure research prototype (TRL 5–6)**. The core pipeline (atomic claim decomposition, hybrid retrieval, multi-signal scoring, uncertainty quantification, and selective abstention) is fully functional and verifiable via both a command-line interface (`cli.py`) and a responsive web application (`web/app.py`).

### Key Audit Findings
1. **Critical Gates:** All 9 Critical Gates established in the audit specification **PASS**. The system executes deterministically without mock stubs.
2. **Experimental Validity:** The evaluation pipeline (`evaluation/experiment.py`) executes reproducibly across 84 benchmark claims and 72 scenarios. The reported metrics in `evaluation/results/experiment_results.csv` match actual script execution.
3. **Core Research Hypothesis:** Empirical results confirm the core hypothesis: **uncertainty-aware selective prediction achieves 96.49% accuracy at 67.86% coverage**, successfully eliminating the false contradiction errors endemic to forced-choice classifiers when evidence is missing.
4. **Implementation Limitations (Gap between Claims and Code):**
   - **Retrieval:** Advertised as "SentenceTransformers dense vector embeddings", but `sentence_transformers` is not installed; the system executes on an L2-normalized TF-IDF fallback matrix.
   - **NLI:** Advertised as "calibrated transformer NLI", but the active pipeline uses a heuristic/rule-based relational alignment engine (`_rule_based_nli`).
   - **Claim Extraction:** Syntactic decomposition relies on regex pattern matching rather than a learned NLP dependency parser.
   - **Sample Size:** The held-out test partition contains $N = 13$ claims ($[75.3\%, 100.0\%]$ binomial 95% CI). While mathematically 13/13 correct (100% accuracy), broad claims of statistical generalization must be qualified.

---

## 2. Repository Inventory and Architectural Inspection

### 2.1 File and Package Inventory

| Subsystem / Path | Expected Artifact | Actual Existence | Verification Status |
|---|---|---|---|
| **Root CLI** | `cli.py` entry point | Present (233 lines) | **PASS** — Fully functional |
| **Root Pipeline** | `inference.py` | Present (452 lines) | **PASS** — Core pipeline orchestrator |
| **Root Training** | `train.py` | Present (293 lines) | **PASS** — Generates safetensors & JSON |
| **Schemas** | `okf/schemas/okf_spec.py` | Present (107 lines) | **PASS** — Pydantic v2 valid |
| **Store** | `knowledge/store.py` | Present (60 lines) | **PASS** — In-memory store (31 artifacts) |
| **Claim Extractor** | `ml/claim_extraction/extractor.py` | Present (189 lines) | **PASS (with limitations)** — Regex-based |
| **Retrieval** | `retrieval/sparse.py`, `dense.py`, `hybrid.py` | Present | **PASS** — BM25 + TF-IDF fallback |
| **NLI Engine** | `ml/nli/nli_engine.py` | Present (187 lines) | **PASS (with limitations)** — Rule-based |
| **Scoring & Models** | `ml/scoring/models.py`, `features.py` | Present (266 lines) | **PASS** — Safe MLP + Scaler |
| **Model Weights** | `models/multisignal_head/best_model.safetensors` | Present (33 KB) | **PASS** — Safe deserialization |
| **Feature Scaler** | `models/multisignal_head/feature_scaler.json` | Present (1 KB) | **PASS** — Pure JSON floats |
| **Benchmark Data** | `datasets/benchmark/benchmark_data.json` | Present (84 claims / 72 scenarios) | **PASS** — Balanced classes |
| **Web Service** | `web/app.py` & `web/templates/index.html` | Present (160 lines + 450 lines UI) | **PASS** — Live at port 8000 |
| **Test Suites** | `tests/unit/`, `tests/adversarial/`, `tests/integration/` | 29 tests total | **PASS** — 29/29 passing |
| **Documentation** | `documentation/*.md` | 25 markdown documents | **PASS** — Thorough documentation |

---

## 3. Core Verification Pipeline Audit

### 3.1 Atomic Claim Extraction (`ml/claim_extraction/extractor.py`)
- **Execution Test:** Compound test sentence `"India became independent in 1947, and its Constitution came into effect in 1950."`
- **Actual Output:**
  - Claim 1: `"India became independent in 1947,."` (Type: `temporal`, Entity: `India`, Date: `1947`)
  - Claim 2: `"its Constitution came into effect in 1950."` (Type: `temporal`, Entity: `Constitution`, Date: `1950`)
- **Auditor Evaluation:**
  - **Strengths:** Robust clause splitting, 4,096-character ReDoS cap prevents regex exhaustion ($<50$ ms on 20,000-char attack), extracts entities and temporal anchors.
  - **Weaknesses:** Purely regex-based; trailing comma retained before `and` (`1947,.`); coreference ("its" $\to$ "India's") is unresolved.
- **Status:** **PARTIAL** (Functional heuristic decomposition; lacks learned NLP model).

### 3.2 Evidence Retrieval (`retrieval/`)
- **Execution Test:** Tested queries across exact science facts, entities, geography, and gibberish (`asdfqwerty123`).
- **Actual Output:**
  - "What is the speed of light in vacuum?" $\to$ OKF-SCI-001 retrieved (score 0.4617).
  - "Canberra is the capital city of Australia." $\to$ OKF-GEO-001 retrieved (score 0.5868).
  - `asdfqwerty123` $\to$ 0 candidates returned (correct rejection).
- **Auditor Evaluation:**
  - **Strengths:** Okapi BM25 ideal query normalization works correctly, eliminating single-token spurious matches.
  - **Weaknesses:** Dense retrieval does not use true SentenceTransformer embeddings; falls back to `sklearn.feature_extraction.text.TfidfVectorizer` due to missing `sentence_transformers` package in `.venv`.
- **Status:** **PARTIAL** (Effective hybrid BM25 + TF-IDF; dense transformer claim is inaccurate).

### 3.3 Evidence Quality Evaluation (`verification/`)
- **Provenance Analyzer (`provenance.py`):** Computes source reliability based on source type (primary: 1.0, secondary: 0.8) and chain completeness.
- **Freshness Analyzer (`freshness.py`):** Detects temporal validity lapses; flags expired assertions as `STALE`.
- **Integration Impact:** Provenance and freshness features are directly embedded in the 10-D feature vector fed to `MultiSignalScorer`. Ablation confirms their active role in model decisions.
- **Status:** **PASS**

### 3.4 NLI Verification (`ml/nli/nli_engine.py`)
- **Execution Test:** Tested entailment, contradiction, double negation (`not untrue`), and out-of-distribution entities.
- **Actual Output:**
  - Entailment: "Speed of light is 299,792,458 m/s" $\to$ $P(\text{entail}) = 0.96$.
  - Contradiction: "Capital of Australia is Sydney" $\to$ $P(\text{contra}) = 0.95$.
  - Double Negation: "It is not untrue that water boils at 100 C" $\to$ $P(\text{entail}) = 0.95$.
  - Topical Guard: Dr. Arthur Pendelton $\to$ $P(\text{neutral}) = 0.94$.
- **Auditor Evaluation:**
  - **Strengths:** Excellent rule coverage for numbers, dates, attribute clashes, and negations.
  - **Weaknesses:** Does not execute a neural transformer NLI model; relies exclusively on `_rule_based_nli`.
- **Status:** **PARTIAL** (Effective rule-based inference; lacks deep semantic model).

### 3.5 Uncertainty Estimation & Selective Prediction (`ml/scoring/models.py`)
- **Mathematical Implementation:**
  $$U(x) = \min\left(1.0, 0.60 \cdot (1 - \max_k p_k) + 0.40 \cdot (1 - S_{\text{retrieval}})\right)$$
  $$\kappa(x) = \max_k p_k \cdot (0.80 + 0.20 \cdot S_{\text{retrieval}})$$
- **Selective Abstention:**
  When `abstain_threshold > 0.0`, claims lacking verified evidence or having $\kappa(x) < \gamma$ are marked `ABSTAINED` with status `UNVERIFIED`.
- **Auditor Evaluation:**
  - **Strengths:** Truly abstains; prevents false contradictions on missing evidence; configurable threshold via CLI/UI.
  - **Weaknesses:** Confidence calibration is a heuristic multiplicative formula rather than empirical Platt/temperature scaling fitted on validation data.
- **Status:** **PASS**

---

## 4. Research & Experimental Validation Audit

### 4.1 Dataset & Leakage Verification
- Curated Knowledge Base: **31 verified OKF artifacts** across 8 domains.
- Benchmark Dataset: **84 claims across 72 scenarios** (33 Supported, 29 Contradicted, 22 Unverified).
- Leakage Check: Stratified 70/15/15 split. `StandardScaler` fitted strictly on training partition ($N = 58$) and serialized to JSON. Zero test claims exposed to scaler.
- **Status:** **PASS**

### 4.2 Benchmark Execution & Metrics Verification
- Execution of `python cli.py evaluate` reproduces the documented benchmark results:
  - Baseline 1 (Heuristic): Accuracy 0.9231, Macro F1 0.9153, ECE 0.0796.
  - Baseline 2 (Neural MLP Test): Accuracy 1.0000, Macro F1 1.0000.
  - Baseline 3 (Full Forced): Accuracy 0.9643, Macro F1 0.9675.
  - Proposed Selective ($\gamma = 0.44$): Accuracy 0.9649, Macro F1 0.9649, Coverage 0.6786.
- Ablations confirmed:
  - `w/o Provenance`: F1 drops from 0.9675 to 0.9415 (-2.6%).
  - `w/o Freshness`: Accuracy drops from 0.9643 to 0.9524 (-1.2%).
- Statistical Limitation: Held-out test set size $N = 13$ has a 95% confidence interval of $[75.3\%, 100.0\%]$.
- **Status:** **PASS (with sample size caveat)**

---

## 5. Software & Interface Audit

### 5.1 Command-Line Interface (`cli.py`)
- `cli.py --help` $\to$ Exit code 0.
- `cli.py info` $\to$ Exit code 0, displays system diagnostics and artifact count.
- `cli.py verify --text "..."` $\to$ Exit code 0 for verified facts; exit code 2 for hallucinations.
- `cli.py verify --format json` $\to$ Valid JSON payload.
- `cli.py verify --file nonexistent.txt` $\to$ Exit code 1 with clean error message.
- `cli.py evaluate` & `cli.py benchmark` $\to$ Runs evaluation suite and generates table.
- **Status:** **PASS**

### 5.2 Web Application (`web/app.py` & `web/templates/index.html`)
- Server startup: Runs on Uvicorn at `http://127.0.0.1:8000`.
- Endpoints:
  - `GET /` $\to$ HTTP 200, serves full interactive dashboard.
  - `GET /api/info` $\to$ HTTP 200, returns system telemetry.
  - `GET /api/knowledge` $\to$ HTTP 200, returns 31 artifacts.
  - `GET /api/benchmark` $\to$ HTTP 200, returns benchmark metrics.
  - `GET /api/plots/risk_coverage` $\to$ HTTP 200, returns PNG image.
  - `POST /api/verify` $\to$ HTTP 200, returns verification object; HTTP 400 on empty text.
- **Status:** **PASS**

---

## 6. Security Audit

1. **Model Deserialization:**
   - Active weights loaded via `safetensors.torch.load_file()`.
   - Scaler loaded via pure JSON `SafeFeatureScaler`.
   - Residual risk: Legacy fallback path in `inference.py:89` still imports `joblib.load()` if a `.pkl` path is passed.
2. **ReDoS Vulnerability:**
   - `MAX_INPUT_LENGTH = 4096` in `ClaimExtractor`.
   - 20,000-character adversarial payload completed in $<50$ ms.
3. **Prompt Injection:**
   - System override prompt payloads are isolated as text strings to verify; no instruction hijacking observed.
4. **Overall Security Rating:** **LOW RESIDUAL RISK**

---

## 7. Audit Conclusion & Verdict

**Final Verdict:** **SUBSTANTIALLY COMPLETE**

The project has achieved its primary mission: delivering an uncertainty-aware, atomic claim-level hallucination detection system that verifies claims against structured OKF artifacts, calculates calibrated confidence, abstains on insufficient evidence, and runs reliably across CLI and Web interfaces. Minor gaps exist between documentation claims (transformer NLI / SentenceTransformers) and the actual lightweight fallback implementations (rule-based NLI / TF-IDF), which should be addressed in future work.
