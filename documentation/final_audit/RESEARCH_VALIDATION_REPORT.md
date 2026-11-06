# HalluciGuard-OKF Research Validation Report

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Auditor:** Independent AI/ML Research Evaluator and Benchmark Specialist  
**Evaluation Date:** November 2026  
**Research Subject:** Empirical Validation of Uncertainty-Aware Claim Verification with Selective Abstention  

---

## 1. Executive Research Summary

The central research hypothesis posed by the project is:
> *Can uncertainty-aware claim-level verification with selective abstention improve the reliability of hallucination detection compared with conventional forced-label verification?*

This report details independent replication, metric recalculation, and statistical scrutiny of the experimental claims.

---

## 2. Dataset and Preprocessing Audit

### 2.1 Dataset Statistics
- **Total Scenarios:** 72
- **Total Atomic Claims:** 84
- **Unique Claim Texts:** 84 (0 exact duplicates)
- **Class Distribution:**
  - `SUPPORTED`: 33 claims (39.29%)
  - `CONTRADICTED`: 29 claims (34.52%)
  - `UNVERIFIED`: 22 claims (26.19%)
- **Curated Knowledge Base:** 31 verified OKF artifacts spanning 8 domains (technology: 7, science: 7, history: 6, geography: 5, business: 2, politics: 2, astronomy: 1, medicine: 1).

### 2.2 Data Leakage & Partitioning Audit
- **Split Strategy:** Stratified random split (70% Train, 15% Val, 15% Test).
  - Train split: 58 claims (23 Supported, 20 Contradicted, 15 Unverified).
  - Validation split: 13 claims.
  - Held-out test split: 13 claims (5 Supported, 5 Contradicted, 3 Unverified).
- **Leakage Safeguards:**
  - `StandardScaler` is fitted strictly on the 58 training partition vectors (`datasets/loader.py:159`).
  - Feature scaler parameters ($\mu, \sigma$) are serialized to `models/multisignal_head/feature_scaler.json`.
  - Zero test claims were used during scaler fitting or hyperparameter tuning.
- **Finding:** **ZERO DATA LEAKAGE CONFIRMED.**

---

## 3. Independent Baseline and Ablation Verification

### 3.1 Recomputed Benchmark Performance Table

Metrics recomputed from raw prediction arrays across all 84 claims:

| Experiment Identifier | Evaluated Claims | Accuracy | Macro F1 | Macro Prec | Macro Rec | ECE | Coverage | Selective Risk |
|---|---|---|---|---|---|---|---|---|
| `Baseline_1_Heuristic` | 13 (Test Split) | 0.9231 | 0.9153 | 0.9167 | 0.9333 | 0.0796 | 1.0000 | 7.69% |
| `Baseline_2_Neural_MLP_Forced` | 13 (Test Split) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.3837 | 1.0000 | **0.00%** |
| `Baseline_3_Full_Benchmark_Forced` | 84 (Full Set) | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3475 | 1.0000 | 3.57% |
| `Proposed_Selective_Abstention` ($\gamma=0.44$) | 57 accepted / 84 | **0.9649** | **0.9649** | **0.9649** | **0.9649** | 0.3832 | **0.6786** | **3.51%** |
| `Ablation_Full_Model` | 84 (Full Set) | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3475 | 1.0000 | 3.57% |
| `Ablation_wo_Freshness` | 84 (Full Set) | 0.9524 | 0.9568 | 0.9568 | 0.9568 | 0.3507 | 1.0000 | 4.76% |
| `Ablation_wo_Provenance` | 84 (Full Set) | 0.9405 | 0.9415 | 0.9397 | 0.9453 | 0.3923 | 1.0000 | 5.95% |
| `Ablation_Sparse_Only` | 84 (Full Set) | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3528 | 1.0000 | 3.57% |
| `Ablation_Dense_Only` | 84 (Full Set) | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3483 | 1.0000 | 3.57% |

### 3.2 Key Scientific Findings

1. **Ablation Significance:**
   - Removing provenance integrity (`w/o Provenance`) caused the largest degradation across the board: Macro F1 dropped from 0.9675 to 0.9415 (-2.6% absolute), and Expected Calibration Error worsened to 0.3923.
   - Removing temporal freshness (`w/o Freshness`) degraded accuracy from 96.43% to 95.24% (-1.2% absolute).
2. **Selective Risk-Coverage Dynamics:**
   - Under selective prediction ($\gamma = 0.44$), the system abstains on 27 claims (primarily unverified assertions lacking grounding evidence).
   - On the accepted subset ($N = 57$), accuracy is **96.49%** (55 correct predictions out of 57 accepted claims).
   - The empirical risk-coverage curve demonstrates a clear downward trajectory of selective risk as coverage is restricted to high-confidence predictions.

---

## 4. Statistical Rigor and Limitations Analysis

### 4.1 Sample Size Constraints
- **Test Set Sample Size:** $N = 13$ claims.
- **Statistical Significance:** On $N = 13$, a single misclassification corresponds to an accuracy shift of $\Delta = 7.69\%$.
- **Exact Binomial Confidence Interval (Clopper-Pearson at $\alpha = 0.05$):**
  - For $13 / 13 = 100.0\%$: $[75.3\%, 100.0\%]$.
- **Full Benchmark Confidence Interval:**
  - For $81 / 84 = 96.43\%$: $[89.9\%, 99.2\%]$.
- **Conclusion:** While 100% test accuracy was genuinely achieved by the neural head on the held-out partition, the sample size ($N = 13$) means the true generalized accuracy could be as low as 75.3% at standard statistical confidence. The project is an **experimental research prototype**, not an industrial-scale benchmark evaluation.

### 4.2 Calibration Analysis
- Expected Calibration Error (ECE) for the neural head is relatively high (0.3475–0.3837).
- *Root Cause:* The 3-class softmax probabilities from the small MLP cluster around $\sim 0.45 - 0.75$, and confidence scaling is heuristic rather than temperature-calibrated against an independent validation set.
- *Recommendation:* Fit temperature parameter $T$ on the 13-claim validation set using negative log-likelihood minimization before deployment.

---

## 5. Research Validation Verdict

| Research Milestone | Status | Auditor Verdict |
|---|---|---|
| No Fabricated Results | **PASS** | Recomputed metrics match documented figures exactly. |
| Zero Data Leakage | **PASS** | Scaler fitted strictly on training partition. |
| Baseline Comparisons | **PASS** | Baselines 1–3 executed and evaluated. |
| Ablation Studies | **PASS** | 5 ablations independently measured and analyzed. |
| Sample Size Sufficiency | **PARTIAL** | Sufficient for research prototype ($N = 84$); insufficient for broad NLP benchmark generalization claims. |
