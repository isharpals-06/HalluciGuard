# INITIAL CODEBASE AUDIT: HALLUCIGUARD-OKF
**Document:** Baseline Repository Audit and State Assessment  
**Author:** Multidisciplinary Engineering and Research Team  
**Date:** November 3, 2026  
**Status:** COMPLETE (Baseline Established)  

---

## 1. Executive Summary of Audit

An exhaustive technical, scientific, and architectural audit of the initial **HalluciGuard-OKF** codebase was completed prior to initiating system improvements. 

The audit established that while the project's documentation ([`README.md`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/README.md), 2,221 lines) outlines an enterprise production platform with FastAPI microservices, PostgreSQL relational models, Celery queues, Redis caching, JWT RBAC security, and a 5-page frontend dashboard, **none of these operational infrastructure components exist in the repository (`NOT IMPLEMENTED`)**.

The actual repository is an early-stage prototype (TRL 2–3) structured as a synchronous CLI Python script with:
1. A 10-dimensional PyTorch MLP classification head ([`models/multisignal_head/best_model.pt`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/best_model.pt)) trained on **19 claims**, validated on **4 claims**, and evaluated on **5 claims**, achieving zero accuracy improvement over a simple 10-line heuristic baseline (80.0% vs. 80.0%) and worse calibration (ECE 0.2412 vs. 0.1865).
2. A deterministic regular-expression claim extractor ([`ml/claim_extraction/extractor.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py)) containing patterns engineered specifically for the README illustrative example.
3. An NLI engine ([`ml/nli/nli_engine.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py)) defaulting to `use_transformer=False`, relying on string heuristics (disjoint 4-digit years, number sets, hardcoded Australian capital city checks, and word overlap).
4. A knowledge store ([`knowledge/artifacts/curated_knowledge.json`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/artifacts/curated_knowledge.json)) limited to 13 curated facts.
5. Insecure deserialization (`torch.load` without `weights_only=True`, `joblib.load`) creating arbitrary code execution vulnerabilities.
6. Execution blocked in the default host Python 3.14 environment due to lack of a dependency lockfile and missing `torch`.

---

## 2. Component Inventory: Working, Broken, and Missing

| Component | File Path | Existing Implementation Status | Audit Assessment |
| :--- | :--- | :--- | :--- |
| **OKF Specification** | [`okf/schemas/okf_spec.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/schemas/okf_spec.py) | **Working (`CONFIRMED`)** | Clean Pydantic v2 schemas modeling provenance, temporal validity, and evidence metadata. |
| **OKF Validator** | [`okf/validator/validator.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/validator/validator.py) | **Working (`CONFIRMED`)** | Validates schemas and temporal chronology (`valid_until >= valid_from`). |
| **OKF Generator** | [`okf/generator/generator.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/generator/generator.py) | **Working (`CONFIRMED`)** | Factory utility for creating valid OKF knowledge objects. |
| **Knowledge Store** | [`knowledge/store.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/store.py) | **Working (`CONFIRMED`)** | In-memory indexing by ID and entity name over 13 static facts. |
| **BM25 Retriever** | [`retrieval/sparse.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/sparse.py) | **Working (`CONFIRMED`)** | Pure-Python BM25 with linear scan ($O(N)$ document scaling). |
| **Dense Retriever** | [`retrieval/dense.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/dense.py) | **Working (Fallback)** | Falls back to Scikit-Learn TF-IDF when `sentence_transformers` is missing. |
| **Hybrid Retriever** | [`retrieval/hybrid.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/hybrid.py) | **Working (`CONFIRMED`)** | Fuses dense and sparse scores with linear weighting and entity boost. |
| **Freshness Evaluator** | [`verification/freshness.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/freshness.py) | **Working (`CONFIRMED`)** | Evaluates temporal validity against reference date and regex keywords. |
| **Provenance Evaluator** | [`verification/provenance.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/provenance.py) | **Working (`CONFIRMED`)** | Weighted formula scoring source reliability and chain completeness. |
| **Claim Extraction** | [`ml/claim_extraction/extractor.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py) | **Fragile (`OBSERVED`)** | Handcrafted regex rules; fails on complex or passive syntax. |
| **NLI Engine** | [`ml/nli/nli_engine.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py) | **Fragile (`OBSERVED`)** | Handcrafted regex/overlap heuristics; lacks real transformer inference. |
| **Feature Extractor** | [`ml/scoring/features.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/features.py) | **Working (`CONFIRMED`)** | Normalizes 10 verification signals into a 10D feature vector. |
| **Scoring Head** | [`ml/scoring/models.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/models.py) | **Fragile (`OBSERVED`)** | PyTorch MLP overfitted to 19 samples; hardcoded override at contradiction > 0.85. |
| **CLI Pipeline** | [`inference.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py) | **Broken on Default Host** | Fails on import without `torch` in host Python environment. |
| **Training Pipeline** | [`train.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/train.py) | **Broken on Default Host** | Fails on import without `torch` in host Python environment. |
| **Test Suite** | [`tests/unit/`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/tests/unit/) | **Partially Broken** | 3 unit tests pass; `test_end_to_end_pipeline.py` fails on import. |
| **Backend REST API** | Documented in README | **NOT IMPLEMENTED** | Directory `backend/` and files do not exist. |
| **Web Frontend** | Documented in README | **NOT IMPLEMENTED** | No frontend dashboard or HTML/JS assets exist. |
| **Database & Cache** | Documented in README | **NOT IMPLEMENTED** | No PostgreSQL, SQLite ORM, or Redis workers exist. |
| **Dependency Lockfile**| Documented in README | **NOT IMPLEMENTED** | No `pyproject.toml` or `requirements.txt` provided. |

---

## 3. Security Weaknesses & CVE-Level Risks

1. **Unsafe Deserialization (VULN-01, CVSS 9.8 - Critical):**
   - [inference.py#L76](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L76): `joblib.load(scaler_path)`
   - [ml/scoring/models.py#L148](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/models.py#L148): `torch.load(path, map_location=self.device)` without `weights_only=True`
   - *Risk:* Arbitrary code execution if model artifacts or scalers are tampered with.
2. **ReDoS Vulnerability (VULN-02, CVSS 7.5 - High):**
   - [ml/claim_extraction/extractor.py#L87-L90](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L87-L90): Nested quantifiers permit exponential backtracking and CPU denial of service on long or repetitive input text.
3. **Unchecked Knowledge Ingestion (VULN-04, CVSS 7.5 - High):**
   - [knowledge/store.py#L23-L37](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/store.py#L23-L37): Lacks digital signature or source authentication, allowing silent knowledge base poisoning.

---

## 4. Dataset Quality & Limitations

- **Curated Knowledge:** Exactly 13 static facts in [`curated_knowledge.json`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/artifacts/curated_knowledge.json).
- **Benchmark Dataset:** Exactly 25 examples comprising 28 claims in [`benchmark_data.json`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/datasets/benchmark/benchmark_data.json).
- **Split Distribution:** Train (19 claims), Validation (4 claims), Test (5 claims).
- **Scientific Implication:** Evaluation on 5 test samples is statistically meaningless (each sample is 20% of the metric, 95% CI is 37.6%–96.4%). The dataset must be expanded and leakage-free splits established.

---

## 5. Reusable Code & Implementation Strategy

### Reusable Core Components:
- The OKF specification and validation engine ([`okf/`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/)).
- The signal feature normalization logic ([`ml/scoring/features.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/features.py)).
- The freshness and provenance analyzers ([`verification/freshness.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/freshness.py), [`verification/provenance.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/provenance.py)).
- Evaluation metric formulas ([`evaluation/metrics.py`](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/evaluation/metrics.py)).

### Actionable Roadmap for Implementation:
1. **Establish Reproducible Environment:** Create a standardized `.venv` using Python 3.11 with `pyproject.toml` and pinned dependencies (`torch`, `pydantic`, `scikit-learn`, `fastapi`, `uvicorn`, `matplotlib`, `safetensors`).
2. **Implement Core Research Pipeline:**
   - NLP-based atomic claim extraction with fallback.
   - Hybrid evidence retrieval with caching and provenance ranking.
   - Genuine NLI verification with transparent fallback labeling.
   - Mathematical uncertainty estimation and selective abstention mechanism.
3. **Dataset Expansion & Rigorous Evaluation:**
   - Expand the benchmark dataset with diverse multi-domain claims.
   - Establish strict leakage-free train/val/test splits.
   - Implement baselines (Heuristic, Neural head, Forced-label) and compare against Proposed Method (Uncertainty-Aware with Selective Abstention).
   - Generate empirical risk-coverage curves and ablation metrics.
4. **Build Unified CLI & Responsive Research Web Application:**
   - Comprehensive CLI supporting verification, evaluation, baselines, and ablations.
   - FastAPI web application delivering claim-level inspection, citations, and risk visualization.
5. **Security Hardening & QA:**
   - Eliminate pickle/joblib vulnerabilities via `safetensors` and JSON configurations.
   - Implement ReDoS safeguards and input limits.
   - Execute full unit, integration, adversarial, and regression test suites.
