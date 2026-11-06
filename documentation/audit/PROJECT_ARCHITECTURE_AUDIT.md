# PROJECT ARCHITECTURE AUDIT: HALLUCIGUARD-OKF

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Target:** Architecture, Directory Structure, Component Implementation, and Execution Flows  
**Auditor Roles:** Senior Software Architect, MLOps & Production Infrastructure Engineer, Technical Auditor  
**Date:** November 3, 2026  
**Status:** COMPLETE  

---

## 1. Directory Structure: Documented vs. Reality

A critical gap exists between the repository structure documented in [README.md#L1166-L1285](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/README.md#L1166-L1285) and the actual filesystem.

### 1.1 Documented Architecture in README
```
Hallucination_detection_system/
├── backend/
│   ├── api/v1/ (analyze.py, claims.py, verify.py, results.py, evidence.py, auth.py, health.py)
│   ├── core/ (config, security, telemetry, database)
│   ├── db/ (models, migrations, repositories)
│   ├── services/ (extractor, retriever, verifier, scorer, explainer)
│   └── workers/ (celery_app.py, tasks.py)
├── frontend/
│   ├── src/ (pages: 1-Input, 2-Overview, 3-Detail, 4-Analytics, 5-KnowledgeAdmin)
│   └── package.json, Dockerfile
├── docker/ (docker-compose.yml, Dockerfile.api, Dockerfile.worker)
├── k8s/ (deployment.yaml, service.yaml, ingress.yaml)
└── tests/ (unit, integration, load, security)
```

### 1.2 Actual Filesystem Structure (`CONFIRMED`)
```
Hallucination_detection_system/
├── configs/
│   ├── model_config.yaml
│   ├── okf_schema.json
│   ├── scoring_thresholds.yaml
│   └── train.yaml
├── datasets/
│   ├── benchmark/
│   │   └── benchmark_data.json
│   ├── loader.py
│   └── __init__.py
├── evaluation/
│   ├── metrics.py
│   └── __init__.py
├── knowledge/
│   ├── artifacts/
│   │   └── curated_knowledge.json
│   ├── store.py
│   └── __init__.py
├── ml/
│   ├── claim_extraction/
│   │   ├── extractor.py
│   │   └── __init__.py
│   ├── nli/
│   │   ├── nli_engine.py
│   │   └── __init__.py
│   ├── scoring/
│   │   ├── features.py
│   │   ├── models.py
│   │   └── __init__.py
│   └── __init__.py
├── models/
│   └── multisignal_head/
│       ├── best_model.pt
│       ├── feature_scaler.pkl
│       └── model_metadata.json
├── notebooks/
│   └── 08_model_training.ipynb
├── okf/
│   ├── generator/
│   │   └── generator.py
│   ├── schemas/
│   │   └── okf_spec.py
│   ├── validator/
│   │   └── validator.py
│   └── __init__.py
├── retrieval/
│   ├── dense.py
│   ├── hybrid.py
│   ├── sparse.py
│   └── __init__.py
├── scripts/
│   └── generate_notebook.py
├── tests/
│   └── unit/
│       ├── test_claim_extraction.py
│       ├── test_end_to_end_pipeline.py
│       ├── test_okf.py
│       └── test_retrieval_and_verification.py
├── verification/
│   ├── freshness.py
│   ├── provenance.py
│   ├── semantic.py
│   └── __init__.py
├── inference.py
├── train.py
└── README.md
```

### 1.3 Architectural Divergence Analysis
- **Missing Backend Subsystem (`NOT IMPLEMENTED`):** Zero API routers, zero FastAPI application instances, zero middleware, zero HTTP request handlers exist.
- **Missing Frontend Subsystem (`NOT IMPLEMENTED`):** No React, Vue, HTML, JavaScript, or CSS assets exist anywhere in the repository.
- **Missing Persistence Subsystem (`NOT IMPLEMENTED`):** No database connection engines (PostgreSQL, SQLite), ORM models (SQLAlchemy, SQLModel), or migration scripts (Alembic) exist.
- **Missing Task Distribution & Caching (`NOT IMPLEMENTED`):** No Celery worker configurations, Redis client handlers, or task queues exist.
- **Missing Containerization & Orchestration (`NOT IMPLEMENTED`):** No `Dockerfile`, `docker-compose.yml`, Helm charts, or Kubernetes YAML manifests exist.
- **Missing Environment & Dependency Manifest (`NOT IMPLEMENTED`):** No `requirements.txt`, `Pipfile`, `poetry.lock`, or `pyproject.toml` exists.

---

## 2. Component Inventory and Status

The table below catalogs every executable Python component and configuration artifact in the repository.

| File Path | Primary Classes / Functions | Responsibility | Status |
| :--- | :--- | :--- | :--- |
| [inference.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py) | `HalluciGuardPipeline`, `main` | Orchestrates the end-to-end detection pipeline from CLI or script. | `CONFIRMED` (Blocked by missing PyTorch) |
| [train.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/train.py) | `train_epoch`, `evaluate_split`, `run_training` | Fits `MultiSignalScorer` MLP on benchmark data, logs baseline, saves artifacts. | `CONFIRMED` (Blocked by missing PyTorch) |
| [configs/model_config.yaml](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/configs/model_config.yaml) | Configuration Dictionary | Declares model dimensions, embedding model name, retrieval alpha. | `OBSERVED` (Partially unreferenced by `inference.py`) |
| [configs/okf_schema.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/configs/okf_schema.json) | JSON Schema Draft 7 | Formal schema specification for OKF artifacts. | `CONFIRMED` |
| [configs/scoring_thresholds.yaml](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/configs/scoring_thresholds.yaml) | Configuration Dictionary | Classification and risk thresholds. | `OBSERVED` (Hardcoded overrides present in code) |
| [configs/train.yaml](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/configs/train.yaml) | Configuration Dictionary | Hyperparameters, optimizer settings, and split ratios for training. | `CONFIRMED` |
| [datasets/loader.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/datasets/loader.py) | `HallucinationDataset`, `BenchmarkDataLoader` | Ingests benchmark JSON, computes signals, splits train/val/test, scales features. | `CONFIRMED` |
| [datasets/benchmark/benchmark_data.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/datasets/benchmark/benchmark_data.json) | Raw JSON list | Benchmark dataset comprising 25 examples and 28 claims. | `CONFIRMED` |
| [evaluation/metrics.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/evaluation/metrics.py) | `compute_calibration_error`, `evaluate_predictions` | Calculates Accuracy, Macro/Weighted F1, ECE, Confusion Matrix. | `CONFIRMED` |
| [knowledge/store.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/store.py) | `OKFKnowledgeStore` | Loads, validates, and indexes OKF objects in memory by ID and entity name. | `CONFIRMED` |
| [knowledge/artifacts/curated_knowledge.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/artifacts/curated_knowledge.json) | Raw JSON list | 13 curated OKF knowledge objects. | `CONFIRMED` |
| [ml/claim_extraction/extractor.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py) | `ExtractedClaim`, `ClaimExtractor` | Decomposes response text into atomic claims via regular expressions. | `CONFIRMED` |
| [ml/nli/nli_engine.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py) | `NLIEngine` | Predicts entailment/neutral/contradiction via regex & lexical heuristics. | `CONFIRMED` |
| [ml/scoring/features.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/features.py) | `SignalFeatureExtractor` | Transforms 10 verification signals into a normalized 10D NumPy vector. | `CONFIRMED` |
| [ml/scoring/models.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/models.py) | `HeuristicThresholdScorer`, `MultiSignalScorer`, `ProductionScorer` | Baseline rule engine, PyTorch residual MLP scorer, and production wrapper. | `CONFIRMED` |
| [models/multisignal_head/best_model.pt](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/best_model.pt) | PyTorch weights binary | Serialized PyTorch weights for `MultiSignalScorer`. | `CONFIRMED` (Trained on 19 samples) |
| [models/multisignal_head/feature_scaler.pkl](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/feature_scaler.pkl) | Pickle binary | Serialized `StandardScaler` from scikit-learn. | `CONFIRMED` |
| [models/multisignal_head/model_metadata.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/model_metadata.json) | JSON file | Training metrics, baseline comparisons, and error analysis samples. | `CONFIRMED` |
| [notebooks/08_model_training.ipynb](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/notebooks/08_model_training.ipynb) | Jupyter Notebook | Synthetic 43-cell experimentation notebook with 0 executed cell outputs. | `OBSERVED` (Unexecuted skeleton) |
| [okf/schemas/okf_spec.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/schemas/okf_spec.py) | `OKFIdentity`, `OKFContent`, `OKFSource`, `OKFProvenance`, `OKFTemporal`, `OKFVerification`, `OKFLifecycle`, `OKFArtifact` | Pydantic v2 schemas defining OKF standard v1.0. | `CONFIRMED` |
| [okf/validator/validator.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/validator/validator.py) | `OKFValidator` | Validates artifacts against Pydantic schema and semantic integrity rules. | `CONFIRMED` |
| [okf/generator/generator.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/generator/generator.py) | `create_okf_artifact` | Factory helper function to instantiate valid OKF artifacts. | `CONFIRMED` |
| [retrieval/sparse.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/sparse.py) | `BM25Retriever` | Pure Python Okapi BM25 implementation over artifact summary text. | `CONFIRMED` |
| [retrieval/dense.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/dense.py) | `DenseRetriever` | Vector retrieval using `sentence-transformers` with TF-IDF fallback. | `CONFIRMED` |
| [retrieval/hybrid.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/hybrid.py) | `HybridRetriever` | Linear score fusion: `alpha * dense + (1 - alpha) * sparse` with entity boost. | `CONFIRMED` |
| [scripts/generate_notebook.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/scripts/generate_notebook.py) | `generate_notebook` | Script that programmatically builds `08_model_training.ipynb`. | `CONFIRMED` |
| [verification/freshness.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/freshness.py) | `FreshnessAnalyzer` | Evaluates temporal validity against reference time and regex indicators. | `CONFIRMED` |
| [verification/provenance.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/provenance.py) | `ProvenanceAnalyzer` | Computes reliability and provenance confidence from metadata weights. | `CONFIRMED` |
| [verification/semantic.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/semantic.py) | `SemanticSimilarityEngine` | Computes embedding cosine similarity with token Jaccard fallback. | `CONFIRMED` |

---

## 3. Actual Technology Stack

| Capability | Documented Stack | Actual Implemented Stack | Audit Assessment |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | Python 3.11+ | Python 3.11 / 3.14 detected | Incompatible: PyTorch unavailable in default Python 3.14 environment. |
| **Data Validation** | Pydantic v2 | Pydantic v2.12.5 | Fully implemented and functional. |
| **Machine Learning / NN** | PyTorch, Transformers | PyTorch (code references only) | Not installed in active runtime environment. |
| **Dense Vector Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Scikit-Learn TF-IDF fallback | `sentence_transformers` is missing; silently defaults to TF-IDF. |
| **Sparse Retrieval** | BM25 (Rank-BM25 / Lucene) | Custom pure-Python BM25 | Pure-Python linear loop, no stemming, no inverted index. |
| **Feature Processing** | Scikit-Learn (`StandardScaler`) | Scikit-Learn 1.9.1 (`StandardScaler`) | Functional, but serialized via unpinned pickle. |
| **Web API Framework** | FastAPI 0.111+, Uvicorn | None (`NOT IMPLEMENTED`) | FastAPI is installed on host, but no endpoints or app file exist. |
| **Database & Cache** | PostgreSQL, Redis, SQLAlchemy | None (`NOT IMPLEMENTED`) | In-memory Python lists and JSON files only. |
| **Testing Framework** | Pytest | Pytest 9.1.1 | 3 test files pass; 1 fails due to `torch` import failure. |

---

## 4. End-to-End Execution Flow (Actual)

The actual runtime execution flow implemented in [inference.py#L78-L304](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L78-L304) proceeds through eight discrete stages:

```
[User Input: Query + Response]
           │
           ▼
[Step 1: Claim Extraction (ClaimExtractor)]
  - Splits text by punctuation (. ! ?)
  - Matches regex pattern r"^(.+?\b(?:was|were)?\s*\w+ed)\s+in\s+(\d{4})\s+by\s+([A-Z][a-zA-Z\s]+)"
  - Outputs ExtractedClaim objects (text, type, entities, extraction_confidence=0.95)
           │
           ▼
[Step 2: Per-Claim Evidence Retrieval (HybridRetriever)]
  - Pure Python BM25 score computed across 13 OKF artifacts
  - Dense cosine similarity computed (SentenceTransformer or TF-IDF fallback)
  - Linear combination: score = 0.6 * dense + 0.4 * sparse (+25% boost if entity matches)
  - Filters candidate against min_relevance_threshold (0.35)
           │
           ├──────────────────────────────────────┐
           ▼ (If score >= 0.35: Evidence Found)   ▼ (If score < 0.35: No Evidence)
[Step 3A: Verification Signals]           [Step 3B: Default Null Signals]
  - Semantic similarity (Jaccard fallback)  - sim_score = 0.02
  - NLI Engine (_rule_based_nli heuristics) - nli_probs = {ent: 0.02, neu: 0.96, con: 0.02}
  - ProvenanceAnalyzer (formulaic weights)  - source_rel = 0.50, prov_conf = 0.20
  - FreshnessAnalyzer (ISO date regexes)    - freshness_state = "UNKNOWN"
           │                                      │
           └──────────────────┬───────────────────┘
                              ▼
[Step 4: Feature Vector Assembly (SignalFeatureExtractor)]
  - Assembles 10D vector: [sim, ent, con, neu, cov, src_rel, prov, fresh, ret, ext]
  - Applies heuristic discount if stale (eff_freshness = 0.3 * fresh, eff_ent = 0.2 * ent)
  - Transforms vector using pre-fitted StandardScaler
                              │
                              ▼
[Step 5: Decision & Classification]
  - IF NOT has_evidence:
      status = "UNVERIFIED", confidence = 0.95, hallucination_score = 0.50
  - ELSE:
      Pass normalized 10D vector to MultiSignalScorer MLP
      Compute class probabilities via Softmax(logits / temperature)
      Compute continuous risk via Sigmoid(risk_head)
      HEURISTIC OVERRIDE: If NLI contradiction > 0.85 -> force CONTRADICTED
                              │
                              ▼
[Step 6: Explanation Synthesis (_build_explanation)]
  - Templates textual explanation based on status, evidence, source, and NLI probabilities
                              │
                              ▼
[Step 7: Response-Level Metric Aggregation]
  - hallucination_rate = contradicted_claims / total_claims
  - evidence_coverage = evidence_found_claims / total_claims
  - IF contradicted > 0: overall_status = "HALLUCINATED", score = 0.40 + 0.60 * rate
  - ELIF all unverified: overall_status = "UNVERIFIED", score = 0.50
  - ELIF all supported:  overall_status = "VERIFIED", score = 0.05
  - ELSE:                overall_status = "PARTIALLY_VERIFIED"
                              │
                              ▼
[Step 8: Output Delivery]
  - Emits JSON payload to stdout or saves to --output path
```

---

## 5. Architectural Bottlenecks & Code Quality Observations

1. **Tight Coupling to Specific Phrasings (`CONFIRMED`):**
   In [ml/claim_extraction/extractor.py#L87-L98](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L87-L98), the sentence decomposition pattern is tightly bound to `<Verb> in <Year> by <Agent>`. Any variation in phrasing (*"In 2008, Elon Musk founded Tesla"*, *"Elon Musk was the founder of Tesla in 2008"*, or *"Tesla was established by Elon Musk in 2008"*) bypasses this decomposition rule completely, leaving the sentence un-decomposed as a compound claim.
2. **Configuration Drift (`CONFIRMED`):**
   [configs/model_config.yaml#L16](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/configs/model_config.yaml#L16) sets `min_relevance_threshold: 0.20`, but [inference.py#L52](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L52) hardcodes default `min_relevance_threshold = 0.35`. Furthermore, `inference.py` instantiates `NLIEngine()` without passing `configs/model_config.yaml` parameters, causing the engine to ignore `roberta-large-mnli` entirely.
3. **In-Memory O(N) Linear Search Bottleneck (`CONFIRMED`):**
   [retrieval/sparse.py#L49-L65](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/sparse.py#L49-L65) iterates through all documents in Python on every query without an inverted index. While fast for 13 documents, this scales as $O(N \cdot M)$ where $N$ is corpus size and $M$ is query length, making it unusable beyond toy datasets.
4. **Hardcoded Model Overrides (`CONFIRMED`):**
   In [inference.py#L215-L218](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L215-L218), the pipeline overrides the trained neural model whenever `nli_probs['contradiction'] > 0.85`. This demonstrates a lack of confidence in the trained neural head, effectively reverting the decision to a hardcoded rule.

---

## 6. Proposed Production Architecture

To fulfill the project's stated objectives, the architecture should be restructured into four cleanly decoupled service boundaries:

```
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Gateway                           │
│     POST /api/v1/verify | GET /api/v1/health | OpenAPI Spec     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
           ┌─────────────────────┴────────────────────┐
           ▼                                          ▼
┌──────────────────────────────┐          ┌───────────────────────┐
│  Ingestion & Knowledge Store │          │ Verification Engine   │
│  - PostgreSQL (Metadata/Log) │          │ - Async Orchestrator  │
│  - Qdrant/FAISS Vector Store │          │ - Redis Result Cache  │
└──────────────┬───────────────┘          └───────────┬───────────┘
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML Verification Workers                      │
│ 1. Claim Decomposition: spaCy / Stanza / Small LLM (vLLM)       │
│ 2. Hybrid Retrieval: Qdrant Dense + Tantivy BM25                │
│ 3. True NLI: Batch RoBERTa / DeBERTa-v3 Cross-Encoder (ONNX)    │
│ 4. Multi-Signal Calibration: Validated GBDT / Calibrated MLP    │
└─────────────────────────────────────────────────────────────────┘
```
