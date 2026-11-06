# HalluciGuard-OKF Remaining Work and Roadmap

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Auditor:** Independent Technical Auditor  
**Date of Audit:** November 2026  
**Document Purpose:** Prioritized inventory of incomplete, partial, or recommended improvements to transition the prototype to enterprise production.

---

## 1. Prioritized Work Items

### Priority 1: High (Core Scientific & Architectural Realism)

#### 1.1 Ingest Genuine Transformer Embeddings
- **Issue:** `DenseRetriever` relies on an L2-normalized TF-IDF fallback matrix because `sentence-transformers` is missing from the environment.
- **Action Required:**
  1. Add `sentence-transformers>=2.2.0` to `requirements.txt` and `pyproject.toml`.
  2. Download and cache a compact local model (`all-MiniLM-L6-v2`, ~80MB).
  3. Precompute dense embeddings for the OKF knowledge base.
- **Estimated Effort:** 2 days.

#### 1.2 Integrate a Real Transformer NLI Head
- **Issue:** The active NLI engine relies on rule-based keyword matching and regex heuristics (`_rule_based_nli`).
- **Action Required:**
  1. Add a compact cross-encoder or MNLI model (e.g. `cross-encoder/nli-distilroberta-base` or `MoritzLaurer/DeBERTa-v3-xsmall-mnli`, ~150MB).
  2. Implement local CPU batching with temperature scaling.
- **Estimated Effort:** 3 days.

#### 1.3 Learned Atomic Claim Decomposition
- **Issue:** `ClaimExtractor` uses regex-based splitting (`re.split`), leaving trailing punctuation artifacts (`1947,.`) and unresolved pronouns ("its Constitution").
- **Action Required:**
  1. Integrate spaCy (`en_core_web_sm`) or a fine-tuned T5 model for atomic proposition extraction.
  2. Implement neural coreference resolution before clause segmentation.
- **Estimated Effort:** 3 days.

---

### Priority 2: Medium (Engineering & Security Hardening)

#### 2.1 Complete Provider Layer Integration
- **Issue:** `HalluciGuardPipeline.analyze()` initializes `self.provider`, but bypasses it during inference, calling internal components directly.
- **Action Required:**
  1. Refactor `analyze()` to delegate claim extraction and single-claim verification to `self.provider.extract_claims()` and `self.provider.verify_claim()` when `provider_name != "local"`.
- **Estimated Effort:** 1 day.

#### 2.2 Remove Legacy Pickle Fallback in `inference.py`
- **Issue:** Line 89 of `inference.py` contains `self.scaler = joblib.load(scaler_path)`.
- **Action Required:**
  1. Remove `joblib.load()` fallback; strictly enforce `feature_scaler.json` loaded via `SafeFeatureScaler`.
- **Estimated Effort:** 1 hour.

#### 2.3 Empirical Temperature Scaling on Validation Split
- **Issue:** Calibrated confidence uses a heuristic formula ($0.8 + 0.2 \times \text{retrieval\_score}$) rather than true optimization.
- **Action Required:**
  1. Fit temperature parameter $T > 0$ on the 13-claim validation set using negative log-likelihood minimization.
  2. Scale logits by $z / T$ before softmax.
- **Estimated Effort:** 1 day.

---

### Priority 3: Low (Scale & Production Readiness)

#### 3.1 Dataset Scale Expansion
- **Issue:** The current benchmark has 84 claims, and the held-out test split has 13 claims.
- **Action Required:**
  1. Expand benchmark to $\ge 500$ claims using semi-automated claim generation from Wikipedia / FEVER.
  2. Narrow binomial confidence intervals to $<3\%$.
- **Estimated Effort:** 1 week.

#### 3.2 Persistent External Vector Database
- **Issue:** The in-memory OKF store holds 31 artifacts in RAM.
- **Action Required:**
  1. Connect retrieval engine to an external vector database (e.g. Qdrant or Milvus) with HNSW indexing for scaling to $>100,000$ artifacts.
- **Estimated Effort:** 4 days.
