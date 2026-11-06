# HalluciGuard-OKF System Architecture

> Current implementation note (2026-11-05): historical descriptions below may refer to the pre-update design. The active NLI mode is rule-based and active dense retrieval is TF-IDF in the configured environment; “calibrated” and “epistemic uncertainty” should not be read as validated properties. See the implementation update at the end of this document and the current research report.

## 1. Architectural Overview

HalluciGuard-OKF is an uncertainty-aware, atomic claim-level factual verification framework designed to detect hallucinations in large language model (LLM) outputs. The architecture decouples generation from factual verification through a structured, multi-signal pipeline grounded in verified Open Knowledge Format (OKF) schema artifacts.

Unlike conventional RAG wrappers or monolithic LLM-as-a-judge approaches, HalluciGuard-OKF operates as a modular, deterministic-first neuro-symbolic pipeline that computes epistemic uncertainty and performs **selective prediction with abstention** when verification evidence is ambiguous or missing.

```
                      +-----------------------------+
                      |   Unstructured Model Text   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Atomic Claim Decomposition |
                      |    (ReDoS Safe Parser)      |
                      +--------------+--------------+
                                     |
                       [Extracted Atomic Claims]
                                     |
                                     v
                      +-----------------------------+
                      |   Hybrid Retrieval Engine   |
                      |  (Okapi BM25 + Dense TFIDF) |
                      +--------------+--------------+
                                     |
                        [Top-k Ranked OKF Evidence]
                                     |
                                     v
       +------------------------------------------------------------+
       |             Multi-Signal Extraction Framework              |
       |  +------------------------+  +--------------------------+  |
       |  |  Natural Language      |  |  Semantic Similarity     |  |
       |  |  Inference (NLI Engine)|  |  Vector Scoring          |  |
       |  +------------------------+  +--------------------------+  |
       |  +------------------------+  +--------------------------+  |
       |  |  Provenance Integrity  |  |  Temporal Freshness      |  |
       |  |  & Authority Analyzer  |  |  Decay & Stale Auditing  |  |
       |  +------------------------+  +--------------------------+  |
       +-----------------------------+------------------------------+
                                     |
                       [10D Normalized Feature Vector]
                                     |
                                     v
                      +-----------------------------+
                      |  Multi-Signal Neural Head   |
                      |  (PyTorch MLP + SafeTensors)|
                      +--------------+--------------+
                                     |
                      +-----------------------------+
                      | Epistemic Uncertainty &     |
                      | Calibrated Selective Logic  |
                      +--------------+--------------+
                                     |
                +--------------------+--------------------+
                |                                         |
     Confidence >= Threshold                  Confidence < Threshold
     & Evidence Present                       or Evidence Missing
                |                                         |
                v                                         v
       [SUPPORTED / CONTRADICTED]                   [ABSTAINED]
       (Evidence Grounded Verdict)              (Unverified Claim)
```

---

## 2. Core Architectural Components

### 2.1 Atomic Claim Decomposer (`ml/claim_extraction/extractor.py`)
- **Mission:** Transforms unstructured multi-sentence text into independent, testable factual propositions.
- **Syntactic Decomposition:** Segments on coordinating conjunctions (`and`, `but`, `whereas`), relative pronouns (`which`, `who`), and temporal adverbs (`while`, `after`).
- **Entity & Temporal Parsing:** Identifies named entities (prioritizing `PROPER_NOUN`), temporal anchors (`1989`, `1994`), and numerical quantities (`299,792,458`).
- **Safety Boundary:** Strict 4,096-character input cap preventing catastrophic regex backtracking (ReDoS attacks). Execution guaranteed in $<50$ ms.

### 2.2 Open Knowledge Format (OKF) Repository (`okf/`, `knowledge/`)
- **Schema Validation:** Strict Pydantic v2 schemas validating identity, content, provenance chains, temporal bounds, and verifier credentials.
- **Storage Layer (`knowledge/store.py`):** In-memory verified knowledge store indexed across 31 curated, domain-diverse factual artifacts spanning physics, geography, technology, medicine, and history.

### 2.3 Hybrid Retrieval Engine (`retrieval/`)
- **Sparse Lexical Engine (`retrieval/sparse.py`):** Okapi BM25 implementation normalized against theoretical maximum query scores to prevent single-token spurious matches.
- **Dense Semantic Fallback (`retrieval/dense.py`):** Dual-path dense vector retrieval leveraging SentenceTransformers (`all-MiniLM-L6-v2`) when available, backed by an English-stopword-filtered TF-IDF vector space model.
- **Convex Score Fusion (`retrieval/hybrid.py`):**
  $$S_{\text{hybrid}}(c, a) = \alpha \cdot S_{\text{dense}}(c, a) + (1 - \alpha) \cdot S_{\text{sparse}}(c, a)$$
  with $\alpha = 0.60$ and target-entity relevance boosting.

### 2.4 Multi-Signal Extraction Engine (`verification/`, `ml/nli/`)
- **NLI Engine (`ml/nli/nli_engine.py`):** General relation and subject-attribute alignment engine detecting Entailment, Contradiction, and Neutral relationships. Implements double-negation resolution (`not untrue` $\to$ entailment) and strict topical grounding to prevent false clashes on unrelated queries.
- **Provenance Analyzer (`verification/provenance.py`):** Evaluates primary vs secondary source credentials and cryptographic signature validation.
- **Freshness Analyzer (`verification/freshness.py`):** Detects temporal decay and flags obsolete assertions (`STALE`, `HISTORICAL`, `CURRENT`).

### 2.5 Multi-Signal Neural Scoring Head (`ml/scoring/models.py`)
- **Architecture:** PyTorch MLP featuring linear projections $(10 \to 64 \to 32 \to 3)$, Batch Normalization, Dropout ($p = 0.20$), and dual prediction heads:
  1. Multi-class classification logits $(\hat{y} \in \{\text{SUPPORTED}, \text{CONTRADICTED}, \text{UNVERIFIED}\})$.
  2. Continuous risk regression head ($\hat{r} \in [0.0, 1.0]$).
- **Safe Serialization:** Stored as zero-vulnerability `.safetensors` tensors and JSON parameter dictionaries (`feature_scaler.json`), eliminating unsafe pickle/joblib execution.

### 2.6 Uncertainty Estimation & Selective Prediction (`ml/scoring/models.py`)
- **Epistemic Uncertainty:**
  $$U(x) = \min\left(1.0, 0.60 \cdot (1 - \max_k p_k) + 0.40 \cdot (1 - S_{\text{retrieval}})\right)$$
- **Selective Abstention:** Replaces forced-choice classification by issuing `ABSTAINED` whenever calibrated confidence falls below threshold $\gamma$ or verified evidence is unavailable.

---

## 3. Communication and Interface Layer

1. **CLI (`cli.py`):** Complete CLI suite (`verify`, `evaluate`, `benchmark`, `info`) with table formatting and JSON output.
2. **FastAPI Web Service (`web/app.py`):** REST API exposing live verification, knowledge inspection, and benchmark telemetry.
3. **Responsive UI (`web/templates/index.html`):** Single-page vanilla JavaScript/CSS dashboard featuring real-time confidence/uncertainty gauges and dynamic claim drill-down cards.
# Implementation update (2026-11-05)

The runtime now reads `configs/model_config.yaml` for NLI/retrieval/scoring settings, exposes active modes in inference results, processes all retained top-K evidence, and provides a separate `evidence_condition` alongside the legacy `status` field. Mixed high support and contradiction signals yield `CONFLICTING_EVIDENCE`, canonical `claim_status=INSUFFICIENT_EVIDENCE`, and a human-review abstention. The default current modes are `rule_based` NLI and `tfidf` dense fallback. Transformer and SentenceTransformer modes fail visibly if unavailable; they do not silently switch.

