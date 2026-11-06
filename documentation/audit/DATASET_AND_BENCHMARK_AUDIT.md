# DATASET AND BENCHMARK AUDIT: HALLUCIGUARD-OKF

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Target:** Benchmark Data, Ground Truth Reliability, Evaluation Protocol, and Statistical Rigor  
**Auditor Roles:** AI Evaluation and Benchmarking Specialist, NLP Research Engineer, Independent Technical Auditor  
**Date:** November 3, 2026  
**Status:** COMPLETE  

---

## 1. Dataset Inventory & Statistical Analysis

The entire dataset assets of the repository consist of two JSON files:
1. `knowledge/artifacts/curated_knowledge.json`: The reference knowledge store.
2. `datasets/benchmark/benchmark_data.json`: The benchmark dataset used for training, validation, and testing.

### 1.1 Curated Knowledge Store ([knowledge/artifacts/curated_knowledge.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/artifacts/curated_knowledge.json))
- **Total Objects:** Exactly 13 knowledge artifacts (`CONFIRMED`).
- **Entity Domain Distribution:**
  - Business / Technology: 4 objects (Tesla, Elon Musk, Apple, Tim Cook)
  - Politics / Current Events: 2 objects (Rishi Sunak, Keir Starmer)
  - Physical Sciences: 3 objects (Speed of light, Gold atomic number, Earth)
  - History / Space Exploration: 2 objects (WWII in Europe, Apollo 11)
  - Geography: 2 objects (Canberra capital of Australia, Mount Everest)
- **Observations:** The knowledge base covers a tiny, hand-selected cluster of trivia facts specifically curated to align with the benchmark test scenarios.

### 1.2 Benchmark Dataset ([datasets/benchmark/benchmark_data.json](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/datasets/benchmark/benchmark_data.json))
- **Total Scenarios ($N$):** 25 examples (`CONFIRMED`).
- **Total Claims:** 28 discrete claims (`CONFIRMED`).
- **Class Distribution (Claim-Level):**
  - `SUPPORTED`: 13 claims (46.4%)
  - `CONTRADICTED`: 10 claims (35.7%)
  - `UNVERIFIED`: 5 claims (17.9%)
- **Claim Typology Distribution:**
  - `temporal`: 9 claims
  - `numerical`: 7 claims
  - `attribution`: 6 claims
  - `entity`: 3 claims
  - `geographic`: 2 claims
  - `factual`: 1 claim
- **Language Coverage:** 100% English (`en`).
- **Synthetic vs. Human-Generated:** 100% synthetic, hand-authored mock scenarios. None are harvested from real-world open-weights or proprietary LLM generation traces (e.g., Llama-3, GPT-4, Claude-3.5).

---

## 2. Train / Val / Test Partition Analysis

In [train.py#L117-L121](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/train.py#L117-L121) and [datasets/loader.py#L139-L156](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/datasets/loader.py#L139-L156), the dataset is split using stratified ratios: 70% Train, 15% Validation, 15% Test.

```
Total Claims: 28
├── Train Split (70%):      19 claims (Class counts: 9 Supported, 7 Contradicted, 3 Unverified)
├── Validation Split (15%):  4 claims (Class counts: 2 Supported, 1 Contradicted, 1 Unverified)
└── Test Split (15%):        5 claims (Class counts: 2 Supported, 2 Contradicted, 1 Unverified)
```

### 2.1 Critical Scientific Flaws in Split Evaluation

1. **Statistical Insignificance ($N_{\text{test}} = 5$):**
   In the test split ([models/multisignal_head/model_metadata.json#L42-L54](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/model_metadata.json#L42-L54)), each sample accounts for exactly **20% of the overall accuracy**.
   - If 4 out of 5 samples are classified correctly, the recorded accuracy is 80.0%.
   - If 1 more sample is misclassified, the accuracy plummets to 60.0%.
   - The 95% Wilson score confidence interval for an observed accuracy of 80% on $N=5$ ranges from **37.6% to 96.4%**. Reporting metrics to four decimal places (`accuracy: 0.8000`, `macro_f1: 0.8222`) conveys a false illusion of precision.
2. **Evaluation Parity with Zero-Intelligence Heuristic:**
   In [models/multisignal_head/model_metadata.json#L13-L64](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/models/multisignal_head/model_metadata.json#L13-L64), the trained deep neural model (`MultiSignalScorer`) and the simple rule-based baseline (`HeuristicThresholdScorer`) produce **identical confusion matrices on the test set**:
   $$\text{Confusion Matrix} = \begin{bmatrix} 2 & 0 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
   Both models achieved Accuracy = 0.8000, Macro Precision = 0.8889, Macro Recall = 0.8333, and Macro F1 = 0.8222. The neural model demonstrated zero empirical gain over a 10-line heuristic rule while deteriorating calibration error ($\text{ECE} = 0.2412$ vs. $0.1865$).
3. **Data Preprocessing & Leakage:**
   [datasets/loader.py#L158-L162](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/datasets/loader.py#L158-L162) properly fits `StandardScaler` only on `X_train`. However, the underlying samples were authored as matched pairs specifically corresponding to the 13 curated facts, introducing structural domain leakage across splits.

---

## 3. Benchmarking Against Established Scientific Datasets

The repository does not integrate, benchmark against, or evaluate on any standard academic hallucination benchmark (`CONFIRMED`).

| Established Benchmark | Scope & Task Definition | Relevance to HalluciGuard-OKF | Feasibility & Recommendation |
| :--- | :--- | :--- | :--- |
| **FEVER** (Thorne et al., 2018) | 185,000+ Wikipedia claims classified as `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`. | **Direct Match (100%):** Matches the 3-way decision taxonomy (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`). | **Mandatory:** The system should evaluate retrieval and NLI against the FEVER test set to establish credible baseline metrics. |
| **FActScore** (Min et al., 2023) | Evaluates atomic factual claim decomposition and precision in generated biographies. | **Direct Match (100%):** Validates atomic claim extraction and factual accuracy. | **Mandatory:** Essential for validating the atomic claim extraction pipeline beyond handcrafted sentences. |
| **RAGTruth** (Wang et al., 2023) | Hallucination corpus across QA, summarization, and data-to-text in RAG systems with word-level annotations. | **High (90%):** Relevant for verifying faithfulness against retrieved context. | **Recommended:** Test whether the detector can identify unsupported claims in retrieved passages. |
| **HaluEval** (Li et al., 2023) | 35,000 generated responses with paired hallucinatory and faithful texts across QA, dialog, and summarization. | **High (85%):** Useful for answer-level hallucination scoring. | **Recommended:** Benchmark overall response-level hallucination scoring. |
| **TruthfulQA** (Lin et al., 2021) | 817 questions designed to trigger human-like false beliefs and popular misconceptions. | **Medium (60%):** Evaluates internal LLM truthfulness rather than RAG verification. | Secondary importance. |

---

## 4. Evaluation Metrics Audit

The metrics calculated in [evaluation/metrics.py](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/evaluation/metrics.py) were audited against scientific benchmarking standards.

```
Implemented Metrics:
├── Accuracy:               sklearn.metrics.accuracy_score (Implemented)
├── Precision (Macro/Wt):   sklearn.metrics.precision_recall_fscore_support (Implemented)
├── Recall (Macro/Wt):      sklearn.metrics.precision_recall_fscore_support (Implemented)
├── F1-score (Macro/Wt):    sklearn.metrics.precision_recall_fscore_support (Implemented)
├── Confusion Matrix:       sklearn.metrics.confusion_matrix (Implemented)
└── Calibration Error:      Custom 10-bin Expected Calibration Error (ECE) (Implemented)

Missing & Dead-Code Metrics:
├── Brier Score:            Imported in metrics.py line 12, but NEVER CALLED in evaluate_predictions() (Dead Code)
├── AUROC (ROC-AUC):        NOT IMPLEMENTED (Critical for threshold-independent evaluation)
├── AUPRC (PR-AUC):         NOT IMPLEMENTED (Critical for imbalanced hallucination detection)
├── False Positive Rate:    NOT IMPLEMENTED (Not computed or reported separately)
├── False Negative Rate:    NOT IMPLEMENTED (Not computed or reported separately)
├── Claim-level vs. Answer-level Split: NOT IMPLEMENTED (Evaluated only at flat claim level)
└── Inference Latency / Cost: NOT IMPLEMENTED in evaluation scripts
```

### Recommendation
1. Expand benchmark data from 28 claims to a minimum of 2,500 claims sampled from FEVER, FActScore, and real LLM outputs.
2. Remove dead import of `brier_score_loss` and implement proper multi-class Brier scoring, AUROC, and AUPRC.
3. Compute and report claim-level metrics separately from response-level aggregated metrics.
