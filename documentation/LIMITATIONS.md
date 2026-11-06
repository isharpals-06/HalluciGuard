# Limitations and Known Constraints

## 1. Scope and Technical Constraints

While HalluciGuard-OKF delivers an uncertainty-aware research prototype with selective abstention, several scientific and practical constraints must be noted:

---

## 2. Specific Technical Limitations

### 2.1 Closed-World Knowledge Base
- **Description:** The system verifies claims against an internal curated knowledge base of 31 verified OKF artifacts.
- **Implication:** The system cannot verify assertions outside its stored topics (e.g. niche pop-culture or localized events). However, rather than generating false contradictions, the selective abstention module cleanly marks missing evidence as `UNVERIFIED` and selectively abstains.

### 2.2 Heuristic-Augmented Transformer NLI
- **Description:** To ensure lightweight local execution on CPU without multi-gigabyte GPU model downloads, the local NLI engine utilizes syntactic relation alignment augmented by semantic similarity embeddings.
- **Implication:** While robust on factual, numerical, and temporal inversions, subtle pragmatic nuances or complex multi-hop syllogisms may yield neutral probabilities rather than definitive entailment.

### 2.3 Syntactic Claim Decomposition
- **Description:** The atomic claim extractor uses deterministic regex and clause boundary rules.
- **Implication:** Highly convoluted, ungrammatical, or run-on sentences may occasionally produce sub-optimal claim splits. The system guards against this via a strict 4,096-character ReDoS cap and subject propagation heuristics.

### 2.4 Scalability of In-Memory Knowledge Store
- **Description:** Artifacts are held in an in-memory list with TF-IDF/BM25 indexing.
- **Implication:** Suitable for prototypes with $<100,000$ artifacts. Scaling to millions of knowledge artifacts requires transitioning to an external vector database (e.g. Qdrant, Milvus, or FAISS).

### 2.5 English-Language Focus
- **Description:** Current tokenization, stopword filtering, and syntax markers are optimized for English-language text.
