# HalluciGuard-OKF Methodology

## 1. Mathematical and Algorithmic Framework

HalluciGuard-OKF implements an end-to-end multi-stage pipeline designed for granular factual verification. This document outlines the mathematical principles, algorithm formulations, and feature definitions powering each component.

---

## 2. Stage-by-Stage Methodology

### 2.1 Atomic Claim Extraction
The atomic decomposition module segments multi-clause generated responses into independent factual assertions.

```
Input Sentence: "Canberra is the capital of Australia, which was founded in 1913."
Decomposition:
  c1: "Canberra is the capital of Australia" [Entity: Canberra, Australia]
  c2: "Canberra was founded in 1913"         [Temporal: 1913]
```

1. **Clause Segmentation:** Identifies syntactic boundaries using coordination markers (`and`, `but`, `while`) and subordinating relative clauses.
2. **Subject Propagation:** If a trailing sub-clause omits an explicit noun phrase subject (e.g. `"...and was founded in 1913"`), the resolved subject from the antecedent clause is prepended.
3. **ReDoS Complexity Bound:** Linear scan algorithm bounded by:
   $$|T| \le 4096 \implies \mathcal{O}(|T|) \text{ worst-case runtime}$$

---

### 2.2 Hybrid Retrieval Engine

Retrieval balances lexical precision against semantic recall.

#### A. Okapi BM25 Sparse Retrieval
Given a query $Q = \{q_1, \dots, q_m\}$ and document $D$:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^m \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
where $k_1 = 1.5, b = 0.75$. To prevent isolated single-word matches from dominating, scores are normalized against the theoretical ideal query score:
$$\text{Score}_{\text{sparse}}(D, Q) = \min\left(1.0, \frac{\text{Score}_{\text{BM25}}(D, Q)}{\sum_{q \in Q} \text{IDF}(q) \cdot (k_1 + 1)}\right)$$

#### B. Dense Semantic Vector Retrieval
Dense representations are generated using transformer sentence embeddings (or TF-IDF vector projections with English stopword filtering):
$$\text{Score}_{\text{dense}}(D, Q) = \frac{\mathbf{v}_Q \cdot \mathbf{v}_D}{\|\mathbf{v}_Q\| \|\mathbf{v}_D\|}$$

#### C. Score Fusion & Filtering
$$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{Score}_{\text{dense}} + (1 - \alpha) \cdot \text{Score}_{\text{sparse}}$$
with candidate threshold $\tau_{\text{retrieval}} = 0.35$.

---

### 2.3 Multi-Signal Feature Extraction

Every candidate evidence-claim pair is mapped to a 10-dimensional continuous vector $\mathbf{x} \in \mathbb{R}^{10}$:

| Index | Feature Dimension | Description | Value Range |
|---|---|---|---|
| `x[0]` | `semantic_similarity` | Cosine similarity between claim and retrieved evidence statement | $[0.0, 1.0]$ |
| `x[1]` | `nli_entailment` | Probability that evidence entails claim | $[0.0, 1.0]$ |
| `x[2]` | `nli_contradiction` | Probability that evidence contradicts claim | $[0.0, 1.0]$ |
| `x[3]` | `nli_neutral` | Probability that evidence is neutral to claim | $[0.0, 1.0]$ |
| `x[4]` | `evidence_coverage` | Binary indicator whether verified evidence was located above $\tau_{\text{retrieval}}$ | $\{0.0, 1.0\}$ |
| `x[5]` | `source_reliability` | Evaluated authority score of evidence source (primary: 1.0, secondary: 0.7) | $[0.0, 1.0]$ |
| `x[6]` | `provenance_confidence` | Verification integrity of provenance chain | $[0.0, 1.0]$ |
| `x[7]` | `freshness_confidence` | Effective temporal validity score discounted by staleness penalty | $[0.0, 1.0]$ |
| `x[8]` | `retrieval_confidence` | Combined hybrid retrieval score | $[0.0, 1.0]$ |
| `x[9]` | `claim_extraction_confidence` | Syntactic confidence in atomic claim segmentation | $[0.0, 1.0]$ |

---

### 2.4 Multi-Signal Neural Classification Head

The neural scoring head maps normalized $\tilde{\mathbf{x}} \in \mathbb{R}^{10}$ to class logits $\mathbf{z} \in \mathbb{R}^3$ and continuous risk $r \in \mathbb{R}$:
$$\mathbf{h}_1 = \text{Dropout}(\text{ReLU}(\text{BatchNorm}(\mathbf{W}_1 \tilde{\mathbf{x}} + \mathbf{b}_1)))$$
$$\mathbf{h}_2 = \text{Dropout}(\text{ReLU}(\text{BatchNorm}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)))$$
$$\mathbf{z} = \mathbf{W}_{\text{cls}} \mathbf{h}_2 + \mathbf{b}_{\text{cls}}, \quad \hat{r} = \sigma(\mathbf{w}_{\text{reg}}^T \mathbf{h}_2 + b_{\text{reg}})$$
where hidden dimensions are $[64, 32]$ and dropout $p = 0.20$.

The training objective optimizes a joint loss function:
$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\mathbf{z}, y; \mathbf{w}_{\text{class}}) + \lambda \cdot \mathcal{L}_{\text{MSE}}(\hat{r}, r_{\text{target}})$$
where $\mathbf{w}_{\text{class}}$ represents inverse class frequency weights and $\lambda = 0.50$.

---

### 2.5 Epistemic Uncertainty & Selective Abstention

1. **Entropy & Retrieval Penalty:**
   $$U(\mathbf{x}) = \min\left(1.0, 0.60 \cdot (1 - \max_k p_k) + 0.40 \cdot (1 - S_{\text{retrieval}})\right)$$
2. **Confidence Calibration:**
   $$\kappa(\mathbf{x}) = \max_k p_k \cdot (0.80 + 0.20 \cdot S_{\text{retrieval}})$$
3. **Selective Abstention Rule:**
   If $\gamma > 0.0$:
   $$(f, g)(\mathbf{x}) = \begin{cases}
   \arg\max_k p_k & \text{if } \kappa(\mathbf{x}) \ge \gamma \;\land\; \mathbf{x}[4] = 1.0 \\
   \text{ABSTAINED} & \text{otherwise}
   \end{cases}$$
