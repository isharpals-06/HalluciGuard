# IMPLEMENTATION PLAN: HALLUCIGUARD-OKF
**Document:** Target System Architecture, Engineering Roadmap, and Acceptance Plan  
**Author:** NLP and LLM Systems Architect, Senior AI/ML Research Engineer  
**Date:** November 3, 2026  
**Status:** FROZEN & READY FOR EXECUTION  

---

## 1. Final Achievable Architecture

The modernized **HalluciGuard-OKF** architecture transitions the repository from an unverified prototype into a robust, experimentally rigorous, and reproducible research prototype supporting both local execution and optional cloud inference.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Interfaces                                 │
│      Unified Research CLI       │        FastAPI Research Web UI       │
│      (halluciguard / python)    │        (Interactive Inspection)      │
└───────────────────┬─────────────┴──────────────────────┬───────────────┘
                    │                                    │
                    ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Core Verification Pipeline Orchestrator                │
│                         (HalluciGuardPipeline)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼──────────────────────────────┐
    ▼                               ▼                              ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  Atomic Claim Engine │ │ Hybrid Retrieval     │ │ Provider Abstraction │
│  - Syntactic Clause  │ │ - OKF Sparse BM25    │ │ - Local Models       │
│    Decomposition     │ │ - TF-IDF / Dense     │ │ - Cloud LLM Options  │
│  - Entity / Temporal │ │ - Provenance &       │ │   (Gemini, OpenAI,   │
│  - Normalized Schema │ │   Freshness Filters  │ │    Claude via .env)  │
└──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Claim-Evidence NLI & Verification Engine                 │
│      - Pretrained Cross-Encoder NLI / Calibrated Heuristic Engine      │
│      - Multi-Evidence Aggregation (Top-K Consensus & Conflict)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│          Uncertainty Estimation & Selective Abstention (Core Novelty)  │
│      - Uncertainty Score = f(NLI Variance, Evidence Relevance, Fresh) │
│      - Calibrated Confidence Estimation (Temperature-Scaled)           │
│      - Selective Decision Rule:                                        │
│          Confidence >= Tau_accept  --> SUPPORTED / CONTRADICTED        │
│          Confidence < Tau_abstain  --> ABSTAIN (INSUFFICIENT_EVIDENCE) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Attribution, Reporting & Evaluation                   │
│      - Traceable Source Citations & Explanations                       │
│      - Risk-Coverage Curve Computation & Selective Accuracy            │
│      - Empirical Evaluation against Baselines & Ablations              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Reusable Modules and Component Contracts

1. **`okf/schemas/okf_spec.py`:** Retained. Serves as the immutable data contract for Open Knowledge Format objects.
2. **`okf/validator/validator.py`:** Retained and extended. Enforces schema validation and temporal chronology.
3. **`knowledge/store.py`:** Retained and expanded. Stores curated knowledge and dynamic benchmark artifacts.
4. **`retrieval/`:** Refactored. Sparse BM25 and dense representations with normalized score fusion and provenance/freshness weighting.
5. **`ml/claim_extraction/`:** Rebuilt. Syntactic clause decomposition, pronoun coreference handling, and structured proposition extraction with robust fallback.
6. **`ml/nli/`:** Upgraded. Clear separation between genuine cross-encoder transformer NLI and transparent rule-based fallback.
7. **`ml/scoring/`:** Rebuilt. Uncertainty estimation, calibrated risk scoring, and selective abstention decision logic.
8. **`evaluation/`:** Rebuilt. Selective accuracy, risk-coverage curves, Brier score, ECE, and macro F1 across baselines.

---

## 3. Dependency Requirements

A standardized environment is built using `uv` and Python 3.11 with the following pinned dependencies:
- `torch>=2.2.0` (MLP head & inference)
- `pydantic>=2.7.0` (OKF data contract validation)
- `scikit-learn>=1.4.0` (TF-IDF vectorizer, scaler, metrics)
- `fastapi>=0.111.0`, `uvicorn>=0.29.0` (Web application interface)
- `matplotlib>=3.8.0` (Risk-coverage and calibration curve plotting)
- `safetensors>=0.4.0` (Safe model weights serialization)
- `pyyaml>=6.0.1` (Config parsing)
- `pytest>=8.1.0` (Testing and QA)

---

## 4. Acceptance Criteria Checklist

- [ ] **A. Core Functionality:** Multi-claim extraction, hybrid retrieval, provenance tracking, 3-way verification (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`), uncertainty scoring, selective abstention, and traceable attribution.
- [ ] **B. Research Implementation:** Forced-label baseline vs. Uncertainty-aware selective abstention; risk-coverage curve generation; empirical ablation evaluation; honest metrics with confidence intervals.
- [ ] **C. Dataset:** Audited benchmark dataset expanded to 100+ claims across diverse domains; leakage-free train/val/test splits.
- [ ] **D. CLI:** Usable interface supporting `verify`, `evaluate`, `benchmark`, and `ablate` with JSON/CSV export.
- [ ] **E. Web Application:** Modern, responsive FastAPI web UI displaying extracted claims, confidence scores, evidence citations, and abstention reasons.
- [ ] **F. Security:** Safe deserialization (no arbitrary pickle), ReDoS input length caps, isolated untrusted evidence.
- [ ] **G. Quality Assurance:** 100% test pass rate across unit, integration, adversarial, and regression test suites.
- [ ] **H. Documentation:** Complete package with all 23 documents in `documentation/`.
