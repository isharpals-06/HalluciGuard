# Future Work and Research Roadmap

## 1. Roadmap Overview

The development roadmap for HalluciGuard-OKF transitions the research prototype into an enterprise-grade factual assurance framework.

---

## 2. Planned Research Directions

### 2.1 Multi-Hop Evidence Reasoning
- **Objective:** Support claims requiring evidence synthesis across multiple discrete OKF artifacts.
- **Approach:** Graph-based reasoning over entity-relationship edges within the OKF knowledge graph using Graph Neural Networks (GNNs) or path-traversal algorithms.

### 2.2 Learned Atomic Claim Decomposition
- **Objective:** Replace rule-based syntactic clause splitting with a fine-tuned sequence-to-sequence model (e.g., T5 or FLAN-T5) trained on WikiAtomicEdits or ClaimDecomp.
- **Benefit:** Resolves complex coreference resolution, implicit premises, and subordinate discourse structures.

### 2.3 External Distributed Vector Store Integration
- **Objective:** Scale the OKF knowledge base from tens of artifacts to tens of millions.
- **Approach:** Integrate persistent vector indexers (e.g. Qdrant or Milvus) with HNSW indexing and sparse-dense hybrid search.

### 2.4 Active Learning for Selective Prediction
- **Objective:** Dynamically tune the selective prediction threshold $\gamma$ using conformal prediction guarantees.
- **Approach:** Guarantee rigorous finite-sample error rate bounds (e.g. $\mathbb{P}(\text{Error} \mid \text{Accepted}) \le 0.05$) via split conformal prediction.

### 2.5 Multi-Lingual OKF Verification
- **Objective:** Extend verification to non-English languages.
- **Approach:** Integrate multilingual transformer encoders (e.g. XLM-RoBERTa) and multilingual stopword/syntactic tokenizers.
