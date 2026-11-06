# HalluciGuard-OKF Documentation Gap Analysis

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Auditor:** Independent Technical Auditor  
**Date of Audit:** November 2026  
**Subject:** Verification of Technical Claims, Documentation Integrity, and Reality Alignment  

---

## 1. Executive Summary

A critical rule of this audit is:
> *A README describing a feature does not prove that the feature exists. Documentation must never receive PASS solely because a file exists.*

This gap analysis compares claims made in the project's documentation (`README.md`, `ARCHITECTURE.md`, `METHODOLOGY.md`, `MODEL_CARD.md`, and `API.md`) against the actual runnable code.

While the documentation is exceptionally thorough (25 documents spanning all major topics), several technical discrepancies were uncovered between advertised capabilities and actual implementations.

---

## 2. Identified Discrepancies and Inaccuracies

### Gap 1: Advertised "SentenceTransformers Dense Retrieval" vs. Actual TF-IDF Fallback
- **Documented Claim:**
  - `README.md` & `ARCHITECTURE.md`: "Dense Semantic Vector Retrieval Service: Encodes OKF knowledge statements and claims into dense representation spaces using SentenceTransformers (`sentence-transformers/all-MiniLM-L6-v2`)."
- **Actual Implementation:**
  - Neither `requirements.txt` nor `pyproject.toml` includes `sentence-transformers`.
  - In `retrieval/dense.py:34-42`, the import fails and triggers:
    `[DenseRetriever] SentenceTransformer initialization note: No module named 'sentence_transformers'. Utilizing fast TF-IDF embedding fallback.`
  - The active dense retrieval engine is an L2-normalized TF-IDF matrix using `sklearn.feature_extraction.text.TfidfVectorizer`.
- **Auditor Assessment:** **DISCREPANCY.** The documentation should honestly describe the active engine as a TF-IDF vector space model with optional SentenceTransformers support.

---

### Gap 2: Advertised "Transformer NLI Engine" vs. Actual Heuristic Rule Alignment
- **Documented Claim:**
  - `README.md` & `METHODOLOGY.md`: "Natural Language Inference (NLI) Verification Engine: Computes calibrated probabilities for Entailment, Neutral, and Contradiction using fine-tuned transformer cross-encoders."
- **Actual Implementation:**
  - In `inference.py:68`, `NLIEngine()` is initialized with no model name and `use_transformer=False`.
  - The pipeline exclusively executes `_rule_based_nli()` in `ml/nli/nli_engine.py`, which uses word token overlap, number extraction, year comparison, and attribute string matching.
- **Auditor Assessment:** **DISCREPANCY.** While the rule-based engine is effective for benchmark facts, describing it as a "transformer NLI" in technical documents is misleading.

---

### Gap 3: Advertised "Calibrated Confidence" vs. Heuristic Scaling Formula
- **Documented Claim:**
  - `METHODOLOGY.md` & `MODEL_CARD.md`: "Calibrated confidence estimation based on empirical validation distributions."
- **Actual Implementation:**
  - In `ml/scoring/models.py:221`:
    `calibrated_confidence = raw_confidence * (0.8 + 0.2 * retrieval_score)`
  - Confidence calibration is not fitted using Platt scaling, temperature scaling, or isotonic regression on a held-out validation set. It is a heuristic multiplicative formula.
- **Auditor Assessment:** **INACCURACY.** True statistical calibration requires validation data fitting. The term "adjusted confidence" is more technically accurate.

---

### Gap 4: Provider Abstraction Integration Bypassed in Pipeline
- **Documented Claim:**
  - `ARCHITECTURE.md` & `API.md`: "Pluggable provider layer allowing execution across local offline models, Google Gemini, and OpenAI."
- **Actual Implementation:**
  - `ml/providers/` contains well-structured provider classes (`LocalProvider`, `GeminiProvider`, `OpenAIProvider`).
  - In `inference.py:73`, `self.provider = get_provider(provider_name)` is initialized.
  - However, the `analyze()` loop directly invokes `self.claim_extractor.extract_claims()` and `self.scorer.predict_single()`, bypassing the provider's `verify_claim()` method.
- **Auditor Assessment:** **PARTIAL INTEGRATION.** The provider classes exist and are functional, but `inference.py` does not route its verification requests through the provider interface.

---

### Gap 5: Generalization Claims vs. Test Split Sample Size
- **Documented Claim:**
  - `MODEL_CARD.md` & `FINAL_COMPLETION_REPORT.md`: "Held-out test split accuracy: 100.0% (Macro F1: 1.0000) confirming generalization."
- **Actual Implementation:**
  - The held-out test partition comprises only 13 claims (5 Supported, 5 Contradicted, 3 Unverified).
  - The exact 95% Clopper-Pearson binomial confidence interval for 13/13 is $[75.3\%, 100.0\%]$.
- **Auditor Assessment:** **OVERSTATED GENERALIZATION.** While the 100% test accuracy is mathematically accurate on the dataset, claiming it "confirms generalizability" across open-domain NLP without noting the sample size is scientifically unsupported.

---

## 3. Gap Severity Summary Table

| Gap ID | Documented Feature | Actual Code Behavior | Severity | Remediation Required |
|---|---|---|---|---|
| **GAP-01** | SentenceTransformers dense embeddings | Normalized TF-IDF fallback matrix | Medium | Clarify in README and ARCHITECTURE that TF-IDF is the default local fallback. |
| **GAP-02** | Transformer NLI cross-encoder | Rule-based semantic & relational alignment | Medium | Update NLI documentation to describe it as a rule-based relation matcher with optional transformer pipeline. |
| **GAP-03** | Calibrated confidence | Heuristic retrieval-weighted confidence | Low | Clarify that scaling is heuristic; implement temperature scaling in future work. |
| **GAP-04** | Provider abstraction integration | Direct components invoked in `inference.py` | Low | Wire `provider.verify_claim()` into `HalluciGuardPipeline`. |
| **GAP-05** | Broad generalization claims | Test split $N = 13$ claims | Low | Add confidence intervals ($[75.3\%, 100.0\%]$) to Model Card and Evaluation report. |
