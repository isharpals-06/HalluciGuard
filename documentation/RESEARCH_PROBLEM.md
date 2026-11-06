# Research Problem Formulation: Uncertainty-Aware Claim Verification with Selective Abstention

## 1. Background and Motivation

Large Language Models (LLMs) regularly produce plausible yet factually incorrect or ungrounded assertions—a phenomenon widely described as *hallucination*. Existing detection methodologies suffer from three fundamental limitations:

1. **Document-Level Granularity:** Evaluating an entire paragraph as a monolithic block masks localized factual errors, yielding low diagnostic utility and poor explainability.
2. **Forced-Label Fallacy:** Conventional verification engines force a binary or ternary decision ($\text{True}/\text{False}$ or $\text{Supported}/\text{Contradicted}/\text{Neutral}$) even when the retrieval pipeline fails to surface relevant evidence. Under forced-choice classification, missing evidence is frequently conflated with factual contradiction, triggering elevated false positive rates.
3. **Miscalibrated Overconfidence:** Neural classification heads trained on small or imbalanced sets exhibit severe overconfidence, assigning high probability masses to erroneous predictions without quantifying epistemic uncertainty.

---

## 2. Formal Problem Statement

Let an LLM-generated output text be denoted by $T$.

### Step 1: Atomic Decomposition
The text $T$ is decomposed into a set of $M$ discrete atomic factual claims:
$$\mathcal{C} = \{c_1, c_2, \dots, c_M\} = \text{Extract}(T)$$
where each $c_i$ represents a single, non-decomposable proposition asserting an entity-attribute-value relationship $(e_i, a_i, v_i)$.

### Step 2: Evidence Grounding
For each claim $c_i$, a retrieval system queries a verified knowledge repository $\mathcal{K}$ containing validated Open Knowledge Format (OKF) artifacts to retrieve top-$k$ evidence candidates:
$$\mathcal{E}_i = \text{Retrieve}(c_i, \mathcal{K}, k)$$
with associated relevance scores $s_{i, j} \in [0, 1]$.

### Step 3: Multi-Signal Representation
For each candidate pair $(c_i, \mathcal{E}_i)$, a feature extractor maps lexical, semantic, inference, provenance, and freshness signals into a fixed-dimensional vector $\mathbf{x}_i \in \mathbb{R}^{10}$.

### Step 4: Selective Prediction Formulation
In standard classification, a model maps $\mathbf{x}_i \to \hat{y}_i \in \mathcal{Y} = \{\text{SUPPORTED}, \text{CONTRADICTED}, \text{UNVERIFIED}\}$.

Under **selective prediction**, the classifier is augmented with a selection function $g(\mathbf{x}_i) \in \{0, 1\}$ that dictates whether the system outputs a prediction or abstains:
$$(f, g)(\mathbf{x}_i) = \begin{cases}
f(\mathbf{x}_i) & \text{if } g(\mathbf{x}_i) = 1 \text{ (Accepted)} \\
\text{ABSTAIN} & \text{if } g(\mathbf{x}_i) = 0 \text{ (Abstained)}
\end{cases}$$

The selection function is governed by calibrated confidence $\kappa(\mathbf{x}_i)$ and epistemic uncertainty $U(\mathbf{x}_i)$ relative to a user-configurable decision threshold $\gamma$:
$$g(\mathbf{x}_i) = \mathbf{1}\left[\kappa(\mathbf{x}_i) \ge \gamma \;\land\; \text{EvidenceFound}(\mathcal{E}_i)\right]$$

---

## 3. Core Research Question

> **Can uncertainty-aware claim-level verification with selective abstention improve the empirical reliability and precision of hallucination detection compared to conventional forced-label verification?**

### Sub-Hypotheses:
1. **$H_1$ (Risk-Coverage Tradeoff):** As the abstention threshold $\gamma$ increases, the selective risk $\mathcal{R}(g)$ (error rate on accepted predictions) decreases monotonically, enabling high-assurance deployment in mission-critical applications.
2. **$H_2$ (Conflation Prevention):** Decoupling unverified claims from contradicted claims through explicit abstention eliminates false positive contradictions arising from out-of-distribution or missing evidence.
3. **$H_3$ (Multi-Signal Synergy):** Incorporating temporal freshness and source provenance features significantly outperforms naive semantic similarity or isolated NLI inference heads.
