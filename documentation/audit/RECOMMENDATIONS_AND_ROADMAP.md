# RECOMMENDATIONS AND PRIORITIZED ROADMAP: HALLUCIGUARD-OKF

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Target:** Prioritized Engineering, Security, and Scientific Remediation Plan  
**Auditor Roles:** Senior Software Architect, AI/ML Research Scientist, Application Security Engineer, MLOps Engineer  
**Date:** November 3, 2026  
**Status:** COMPLETE  

---

## 1. Prioritization Framework

Recommendations are classified into four strict priority tiers:
- **P0 — Critical:** Security vulnerabilities, runtime crash blockers, and methodological issues that completely invalidate evaluation results.
- **P1 — High Priority:** Core algorithmic weaknesses, missing empirical validation, and primary architectural blockers.
- **P2 — Medium Priority:** Performance optimizations, asynchronous pipeline migration, monitoring, and developer tooling.
- **P3 — Future Research:** Advanced research directions, multimodal extensions, and long-horizon capabilities.

---

## 2. Master Prioritized Action Matrix

---

### Priority Tier P0: Critical (Immediate Resolution Required)

#### REC-01: Remediate Insecure Deserialization Vulnerability (VULN-01)
- **Problem:** Arbitrary code execution via `torch.load()` without `weights_only=True` and unpinned `joblib.load()`.
- **Evidence:** [inference.py#L76](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L76) and [ml/scoring/models.py#L148](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/models.py#L148).
- **Proposed Solution:**
  1. Update `torch.load(path, map_location=self.device, weights_only=True)`.
  2. Transition model weight serialization from `.pt` pickle to Hugging Face `safetensors`.
  3. Replace `feature_scaler.pkl` with a plain JSON or NumPy `.npz` configuration file storing `mean_` and `scale_` arrays.
- **Expected Benefit:** Neutralizes remote code execution attack vector; achieves zero-vulnerability security scan.
- **Implementation Complexity:** Low (1 developer-day).
- **Dependencies:** `safetensors`.
- **Validation Criteria:** Execution of pickle exploit payload throws an exception; model weights load successfully without pickling.
- **Priority:** **P0**

---

#### REC-02: Establish Pinned Dependency Manifest and Resolve Runtime Crash
- **Problem:** Repository lacks `requirements.txt` or `pyproject.toml`. Default environment lacks `torch`, causing `inference.py` and `test_end_to_end_pipeline.py` to crash immediately on import.
- **Evidence:** Root directory inspection and pytest failure output in [tests/unit/test_end_to_end_pipeline.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/tests/unit/test_end_to_end_pipeline.py).
- **Proposed Solution:**
  Create a standardized `pyproject.toml` with pinned dependency specifications compatible with Python 3.11/3.12:
  ```toml
  [project]
  name = "halluciguard-okf"
  version = "1.0.0"
  dependencies = [
      "torch>=2.2.0",
      "transformers>=4.40.0",
      "sentence-transformers>=2.7.0",
      "pydantic>=2.7.0",
      "scikit-learn>=1.4.0",
      "fastapi>=0.111.0",
      "pyyaml>=6.0.1",
      "pytest>=8.1.0"
  ]
  ```
- **Expected Benefit:** Ensures 100% reproducible installation and allows `test_end_to_end_pipeline.py` to pass without import failures.
- **Implementation Complexity:** Low (2 hours).
- **Dependencies:** None.
- **Validation Criteria:** `pytest tests/unit` executes and passes 100% of unit and integration tests cleanly.
- **Priority:** **P0**

---

#### REC-03: Remediate ReDoS Vulnerabilities in Claim Extraction & NLI
- **Problem:** Catastrophic backtracking in nested regular expressions allows denial of service via long or repetitive input text.
- **Evidence:** [ml/claim_extraction/extractor.py#L87-L90](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L87-L90), [ml/claim_extraction/extractor.py#L164](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L164).
- **Proposed Solution:**
  1. Enforce strict character limits on input responses (maximum 4,096 characters per API call).
  2. Replace compound regexes with linear string operations or compile regexes using Google's `re2` engine.
- **Expected Benefit:** Eliminates denial-of-service attack surface; guarantees linear-time preprocessing.
- **Implementation Complexity:** Low (1 developer-day).
- **Dependencies:** `google-re2`.
- **Validation Criteria:** Submitting a 10,000-character adversarial string completes in $< 25$ ms without CPU freeze.
- **Priority:** **P0**

---

### Priority Tier P1: High Priority (Scientific & Architectural Grounding)

#### REC-04: Replace Regex Heuristics with Genuine Transformer NLI Models
- **Problem:** Production inference defaults to `use_transformer=False` and relies on naive word overlap and hardcoded Australian capital city checks.
- **Evidence:** [ml/nli/nli_engine.py#L18-L41](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py#L18-L41), [ml/nli/nli_engine.py#L97-L107](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py#L97-L107).
- **Proposed Solution:**
  Enable transformer-based cross-encoder inference using a compact, pre-trained NLI model (e.g. `cross-encoder/nli-deberta-v3-small` or `roberta-large-mnli`). Export model to ONNX runtime for sub-40 ms CPU inference.
- **Expected Benefit:** Resolves paraphrase failure modes; eliminates brittle regex rules; provides scientifically valid entailment probabilities.
- **Implementation Complexity:** Medium (3 developer-days).
- **Dependencies:** `transformers`, `onnxruntime`.
- **Validation Criteria:** Successfully identifies semantic paraphrases and double negations on standard ANLI benchmark examples.
- **Priority:** **P1**

---

#### REC-05: Replace Handcrafted Regex Claim Extraction with Robust NLP Parser
- **Problem:** Custom regexes fail on non-standard phrasing and complex syntax.
- **Evidence:** [ml/claim_extraction/extractor.py#L87-L98](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L87-L98).
- **Proposed Solution:**
  Integrate an industrial NLP dependency parser (spaCy `en_core_web_sm` / Stanza) or an open-weight instruction-tuned LLM (Qwen-2.5-3B via vLLM) using structured JSON output to decompose complex sentences into atomic claims while maintaining coreference resolution.
- **Expected Benefit:** Extracts atomic claims reliably across arbitrary syntactic structures, relative clauses, and passive voice.
- **Implementation Complexity:** Medium (4 developer-days).
- **Dependencies:** `spacy` or `vllm`.
- **Validation Criteria:** Achieves $> 85\%$ claim decomposition accuracy on FActScore benchmark.
- **Priority:** **P1**

---

#### REC-06: Scale Benchmark Dataset Beyond 28 Toy Claims
- **Problem:** Model is trained on 19 samples, validated on 4, and tested on 5, rendering all reported evaluation metrics statistically invalid.
- **Evidence:** [models/multisignal_head/model_metadata.json#L42-L54](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/model_metadata.json#L42-L54).
- **Proposed Solution:**
  Harvest and format a minimum of 2,500 real-world claim-evidence pairs from established academic datasets: FEVER (fact-checking), FActScore (biography hallucinations), and RAGTruth (retrieval-augmented hallucinations).
- **Expected Benefit:** Yields statistically rigorous, publishable evaluation results with meaningful confidence intervals.
- **Implementation Complexity:** Medium (3 developer-days).
- **Dependencies:** `datasets` (Hugging Face).
- **Validation Criteria:** Evaluation report provides Macro F1, AUROC, and ECE with 95% bootstrap confidence intervals across $\ge 500$ held-out test claims.
- **Priority:** **P1**

---

### Priority Tier P2: Medium Priority (Production Infrastructure & Engineering)

#### REC-07: Implement Documented FastAPI Endpoints
- **Problem:** All documented REST endpoints (`/api/v1/analyze`, `/api/v1/verify`, `/health`) are non-existent.
- **Evidence:** Absence of `backend/` directory in repository.
- **Proposed Solution:**
  Implement a production FastAPI application in `backend/main.py` exposing asynchronous endpoints with Pydantic request/response validation, OpenTelemetry instrumentation, and Swagger UI documentation.
- **Expected Benefit:** Enables integration with upstream microservices, frontend dashboards, and client SDKs.
- **Implementation Complexity:** Medium (4 developer-days).
- **Dependencies:** `fastapi`, `uvicorn`.
- **Validation Criteria:** Automated integration tests successfully query `/api/v1/verify` via `httpx.AsyncClient`.
- **Priority:** **P2**

---

#### REC-08: Migrate Vector Retrieval from In-Memory Scan to Qdrant/FAISS
- **Problem:** BM25 and dense retrieval execute $O(N)$ linear scans in pure Python, causing severe performance degradation as corpus size grows.
- **Evidence:** [retrieval/sparse.py#L48-L65](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/sparse.py#L48-L65).
- **Proposed Solution:**
  Deploy an embedded vector database (e.g. Qdrant or FAISS HNSW) with a Tantivy full-text index for hybrid lexical-semantic retrieval.
- **Expected Benefit:** Constant $O(\log N)$ retrieval latency ($< 10$ ms) over corpora exceeding 500,000 documents.
- **Implementation Complexity:** Medium (3 developer-days).
- **Dependencies:** `qdrant-client` or `faiss-cpu`.
- **Validation Criteria:** Query latency remains below 15 ms on a test corpus of 100,000 OKF documents.
- **Priority:** **P2**

---

### Priority Tier P3: Future Research (Advanced Capabilities)

#### REC-09: Implement Temporal-Constraint NLI (Research Direction 2)
- **Problem:** Cannot resolve evolving temporal states (e.g. past vs. current leadership).
- **Proposed Solution:** Integrate temporal expression tagging with interval algebra into cross-encoder attention matrices.
- **Priority:** **P3**

#### REC-10: Implement Conformal Selective Abstention (Research Direction 5)
- **Problem:** System lacks mathematical guarantees on false positive rates under open-domain missing knowledge.
- **Proposed Solution:** Deploy conformal prediction calibration to adaptively abstain from classification when uncertainty bounds are exceeded.
- **Priority:** **P3**
