# AI/ML METHODOLOGY AUDIT: HALLUCIGUARD-OKF

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Target:** Mathematical, Algorithmic, and Scientific Validity of Detection Logic  
**Auditor Roles:** Senior AI/ML Research Scientist, LLM & Transformer Architect, NLP Research Engineer  
**Date:** November 3, 2026  
**Status:** COMPLETE  

---

## 1. Hallucination Classification Coverage

The system defines hallucination primarily as **direct contradiction between an atomic claim and retrieved reference knowledge** (`CONTRADICTED`), while treating lack of reference knowledge as `UNVERIFIED`. 

Below is an evidence-based audit of how the implementation handles the 10 core hallucination taxonomies.

| Hallucination Type | Implementation Status | Technical Mechanism & Evidence | Limitations & Failure Modes |
| :--- | :--- | :--- | :--- |
| **1. Factual Hallucinations** | **Partially Supported** | Matches claim against 13 static facts in [knowledge/artifacts/curated_knowledge.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/artifacts/curated_knowledge.json) using BM25/Cosine. | Fails for any fact outside the 13 curated objects. Cannot verify open-domain world knowledge. |
| **2. Contextual Hallucinations** | **Unsupported (`NOT IMPLEMENTED`)** | The system inspects claims in isolation. Does not ingest prior conversational history or system prompts. | Cannot detect if an answer violates earlier conversation context or prompt constraints. |
| **3. Faithfulness Violations** | **Unsupported (`NOT IMPLEMENTED`)** | The pipeline only compares responses against its internal OKF store, not against user-supplied context documents. | In RAG use cases, cannot determine whether an LLM answered faithfully from the prompt's provided context. |
| **4. Unsupported Claims** | **Supported** | Handled via `UNVERIFIED` state ([inference.py#L201-L206](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L201-L206)) when retrieval relevance score is below 0.35. | Does not verify whether a claim is inherently unverifiable or simply missing from its local JSON store. |
| **5. Fabricated Citations** | **Unsupported (`NOT IMPLEMENTED`)** | No citation verification logic, DOI lookup, PubMed resolution, or URL health check exists. | Completely blind to fabricated academic papers, fake URLs, or hallucinated court cases. |
| **6. Entity Hallucinations** | **Partially Supported** | Extracts capitalized noun tokens via regex ([ml/claim_extraction/extractor.py#L164-L168](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L164-L168)). | Fails on lowercase entities, acronyms, or non-English entities; treats any capitalized word as an entity. |
| **7. Numerical Hallucinations** | **Partially Supported (Brittle)** | Regex set comparison in [ml/nli/nli_engine.py#L86-L91](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py#L86-L91) flags contradiction if number sets are disjoint. | Fails on equivalent numerical representations ($10^6$ vs 1,000,000, 50% vs 0.5), ranges, or calculated ratios. |
| **8. Temporal Hallucinations** | **Partially Supported (Brittle)** | Regex extraction of 4-digit years (1700–2099) in [ml/nli/nli_engine.py#L76-L84](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py#L76-L84) and stale state detection in [verification/freshness.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/freshness.py). | Fails on relative temporal expressions ("last Tuesday", "three decades ago", "during the Bronze Age"). |
| **9. Logical Inconsistencies** | **Unsupported (`NOT IMPLEMENTED`)** | No formal logic engine, SAT solver, or syllogistic verification exists. | Cannot detect internal reasoning contradictions across sentences (e.g., "A > B, B > C, therefore C > A"). |
| **10. Partial Hallucinations** | **Partially Supported** | Sentence splitting attempts to isolate sub-claims ([ml/claim_extraction/extractor.py#L78-L121](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L78-L121)). | Dependent on brittle regexes. If compound sentence syntax deviates from `<Verb> in <Year> by <Agent>`, it fails to split. |

---

## 2. Detection Methodology Deep Dive

### 2.1 Natural Language Inference (NLI)
- **What it Actually Executes (`CONFIRMED`):**
  Despite configuration files citing `roberta-large-mnli` ([configs/model_config.yaml#L19](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/configs/model_config.yaml#L19)), the system runtime defaults to `use_transformer=False` ([ml/nli/nli_engine.py#L18-L24](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py#L18-L24)). The active engine is `_rule_based_nli`, which executes handcrafted heuristics:
  1. Compares sets of 4-digit years: disjoint sets yield `P(contradiction) = 0.94`.
  2. Compares numbers: disjoint number sets yield `P(contradiction) = 0.90`.
  3. Checks Australian capital cities with hardcoded regex: if hypothesis mentions Sydney and premise Canberra, returns `P(contradiction) = 0.94` ([lines 97-107](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/nli/nli_engine.py#L97-L107)).
  4. Checks "by [Name]" agentive phrases: disjoint sets return `P(contradiction) = 0.92`.
  5. Computes lexical token overlap ratio: if overlap $\ge 0.70$, $P(\text{entailment}) = \min(0.98, 0.45 + 0.55 \cdot \text{overlap})$.
- **Assumptions:** Assumes that factual contradiction is strictly synonymous with mismatching integers, years, or negation markers, and that entailment equals high bag-of-words overlap.
- **Critical Failure Modes:**
  - Synonymy and Paraphrasing: *"The vehicle manufacturer was initiated by Martin Eberhard"* fails overlap check ($\text{overlap} < 0.40$), yielding false Neutral/Unverified.
  - Bag-of-Words Entailment Failure: *"The dog bit the man"* and *"The man bit the dog"* have 100% token overlap, incorrectly producing $P(\text{entailment}) = 0.98$.
- **Computational Cost:** Extremely low ($< 0.5$ ms, pure CPU string ops), but scientifically invalid for generalized NLI.

### 2.2 Semantic Similarity & Dense Retrieval
- **What it Actually Executes (`CONFIRMED`):**
  In [retrieval/dense.py#L35-L52](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/retrieval/dense.py#L35-L52), `DenseRetriever` attempts to load `sentence-transformers/all-MiniLM-L6-v2`. Because `sentence_transformers` is not installed in the environment, it silently activates the fallback: a Scikit-Learn `TfidfVectorizer(ngram_range=(1, 2), max_features=512)`.
  In [verification/semantic.py#L33-L44](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/verification/semantic.py#L33-L44), when dense retriever is uninitialized, it falls back to token Jaccard similarity:
  $$\text{sim} = \min\left(1.0, \frac{|T_{\text{claim}} \cap T_{\text{evidence}}|}{|T_{\text{claim}} \cup T_{\text{evidence}}|} \times 1.5\right)$$
- **Critical Failure Modes:**
  - Inability to recognize semantic equivalence across differing vocabularies (e.g., *"lunar module"* vs. *"spacecraft landed on the moon"*).
  - Susceptibility to lexical distraction: two sentences sharing common terminology on different topics receive artificially high similarity.

### 2.3 Atomic Claim Extraction
- **What it Actually Executes (`CONFIRMED`):**
  [ml/claim_extraction/extractor.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py) uses regex splitting on punctuation and conjunctions (`and`, `but`, `however`, `while`). Lines 87–98 contain a specific regex:
  ```python
  temp_by_pattern = re.match(
      r"^(?P<subj>.+?\b(?:was|were|is|are)?\s*\w+ed)\s+in\s+(?P<year>1[789]\d{2}|20\d{2})\s+by\s+(?P<agent>[A-Z][a-zA-Z\s]+)[.]?$",
      sentence.strip(),
  )
  ```
- **Audit Finding:** This regex matches the exact phrasing of the illustrative example in the README: *"Tesla was founded in 2008 by Elon Musk."*
- **Failure Modes:**
  - Fails to extract claims from relative clauses (*"The company, which Elon Musk led after 2004, manufactured EVs"*).
  - Fails on complex conjunctions, conditional statements, or subordinate clauses.
  - Drops semantic context from coreference: *"It is a transition metal"* cannot resolve *"It"* back to *"Gold"*.

### 2.4 Multi-Signal Scoring and Calibration Head
- **What it Actually Executes (`CONFIRMED`):**
  [ml/scoring/models.py#L66-L130](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/models.py#L66-L130) defines a PyTorch neural network:
  - Input: 10 hand-crafted features.
  - Architecture: Linear(10, 64) $\to$ LayerNorm $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(64, 32) + Skip Connection(Linear(10, 32)) $\to$ LayerNorm $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(32, 3) classification head + Linear(32, 1) continuous risk head.
  - Output: Scaled by trainable temperature parameter $\tau \in [0.1, 5.0]$.
- **Methodological Evaluation:**
  - **Severe Overfitting Risk:** The model has 3,748 trainable parameters, but was trained on **19 training instances** ([models/multisignal_head/model_metadata.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/model_metadata.json)). Training a 3,700-parameter deep network on 19 samples violates fundamental statistical learning principles (parameter-to-sample ratio > 190:1).
  - **Zero Metric Superiority:** On the 5-sample test set, the neural model achieves the exact same accuracy (0.8000) and macro F1 (0.8222) as the trivial heuristic threshold baseline `HeuristicThresholdScorer`, but produces **worse calibration error** ($\text{ECE} = 0.2412$ vs. $0.1865$).

---

## 3. Reliability Analysis: 4-Way Decision Separation

A core requirement is distinguishing between `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT EVIDENCE` (`UNVERIFIED`), and `UNCERTAIN`.

```
Decision Matrix Audit:
┌────────────────────────┬───────────────────────────────────────┬───────────────────────────────┐
│ System Decision State  │ Code Implementation Mechanism          │ Reliability Assessment        │
├────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ SUPPORTED              │ Model Class 0 OR (Entailment >= 0.55  │ Reliable ONLY if evidence is  │
│                        │ AND Sim >= 0.30)                      │ retrieved; fails on paraphrase│
├────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ CONTRADICTED           │ Model Class 1 OR Contradiction > 0.85 │ Highly brittle; triggers on   │
│                        │ OR Freshness == STALE                 │ year/number string mismatch   │
├────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ UNVERIFIED             │ Retrieval score < 0.35 OR             │ Conceptually sound policy;    │
│ (Insufficient Evidence)│ Ambiguous NLI / Low confidence        │ vulnerable to retrieval miss  │
├────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ UNCERTAIN              │ Dynamic flag 'LOW_CONFIDENCE' added   │ Partially supported as a flag,│
│                        │ if max class probability < 0.60       │ not a distinct class state    │
└────────────────────────┴───────────────────────────────────────┴───────────────────────────────┘
```

### Critical Finding on "Absence of Evidence"
The system successfully enforces the rule that **absence of evidence is not proof of hallucination**. In [inference.py#L201-L206](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L201-L206):
```python
if not has_evidence:
    # Absence of evidence is strictly UNVERIFIED per Section 2
    status = "UNVERIFIED"
    conf = 0.95
    flags = ["NO_EVIDENCE"]
    hallucination_score = 0.50
```
However, because retrieval relies on an in-memory TF-IDF fallback over only 13 documents, true factual statements (e.g., *"Water boils at 100°C"*) fail retrieval and are classified as `UNVERIFIED` with high confidence ($0.95$), leading to massive false-negative rates for open-domain factual verification.
