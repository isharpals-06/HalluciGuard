# Experimental Design and Protocol

## 1. Experimental Methodology

The experimental framework evaluates the primary research hypothesis:
> *Can uncertainty-aware claim-level verification with selective abstention improve the empirical reliability and precision of hallucination detection compared to conventional forced-label verification?*

All experiments are conducted using reproducible, deterministic protocols with fixed random seeds (`seed = 42`).

---

## 2. Compared Models and Baselines

### 2.1 Baseline 1: Heuristic Rule-Based Scorer
- **Description:** Standard industry heuristic baseline.
- **Decision Logic:** Employs fixed threshold logic over lexical similarity ($S_{\text{sim}} \ge 0.70$) and heuristic NLI entailment scores ($P_{\text{entail}} \ge 0.65$) to assign `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIED`.
- **Operating Mode:** Forced-choice classification without calibrated confidence or uncertainty-aware abstention.

### 2.2 Baseline 2: Restored Neural MLP Head (Held-out Test Split)
- **Description:** The trained multi-layer perceptron architecture evaluated on the isolated 15% held-out test split ($N = 13$).
- **Features:** Normalized 10-dimensional feature vector.
- **Operating Mode:** Forced-choice prediction ($\gamma = 0.0$).

### 2.3 Baseline 3: Forced-Label Verification (Full Benchmark)
- **Description:** Neural MLP head evaluated across the complete benchmark dataset ($N = 84$) with abstention disabled ($\gamma = 0.0$).
- **Operating Mode:** Forces a categorical verdict on every claim, regardless of retrieval quality or prediction confidence.

### 2.4 Proposed Method: Uncertainty-Aware Selective Prediction
- **Description:** The complete HalluciGuard-OKF pipeline incorporating calibrated confidence estimation and selective abstention ($\gamma = 0.44$).
- **Selection Function:** Claims with calibrated confidence below $\gamma$ or lacking verified evidence are abstained (`ABSTAINED`).
- **Evaluation:** Evaluated on selective risk, selective accuracy, coverage rate, and Expected Calibration Error (ECE).

---

## 3. Evaluation Metrics

1. **Classification Accuracy & Macro F1:**
   $$\text{Macro F1} = \frac{1}{|\mathcal{Y}|} \sum_{c \in \mathcal{Y}} F_{1, c}$$
2. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$
   evaluated across $M = 10$ uniform confidence bins.
3. **Selective Risk & Coverage:**
   $$\text{Coverage}(g) = \frac{1}{N} \sum_{i=1}^N g(\mathbf{x}_i)$$
   $$\text{Selective Risk}(f, g) = \frac{\sum_{i=1}^N \mathbf{1}[f(\mathbf{x}_i) \neq y_i] \cdot g(\mathbf{x}_i)}{\sum_{i=1}^N g(\mathbf{x}_i)}$$
4. **Risk-Coverage Curve:** Empirical curve tracing selective risk as coverage is swept from 10% to 100%.

---

## 4. Ablation Study Protocol

To isolate the marginal contribution of each architectural subsystem, the full benchmark is evaluated under five controlled ablation configurations:
1. **Full Model:** All 10 signal dimensions active.
2. **w/o Freshness Analysis:** Temporal validity and staleness features zeroed out ($x_7 = 0.0$).
3. **w/o Provenance Integrity:** Source authority and provenance verification confidence zeroed out ($x_5 = x_6 = 0.0$).
4. **Sparse-Only Retrieval:** Dense semantic vector similarity removed ($x_0 = 0.0$).
5. **Dense-Only Retrieval:** Sparse BM25 lexical contribution suppressed ($x_8 = x_0$).
# Current protocol status (2026-11-05)

The historical protocol/results below do not describe the current training pipeline. Current training derives evidence features from the local retrieval store and groups examples by scenario. It records validation-fitted temperature scaling and validation-selected selective-risk threshold, then evaluates on the 14-claim held-out test partition. The latest actual metrics and limitations are in `documentation/RESEARCH_REPORT.md` and the current run bundle.

The deterministic perturbation utility exists in `evaluation/perturbations.py`, but no full perturbation experiment has yet been run. Do not treat the earlier ablation or selective-prediction numbers below as current findings.

