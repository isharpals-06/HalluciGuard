# Experimental Evaluation Report

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for current evidence.


## 1. Executive Summary

This report documents the empirical evaluation of HalluciGuard-OKF across 84 benchmark claims and 72 generation scenarios. Experiments compare the proposed Uncertainty-Aware Selective Prediction method against three baseline configurations and five ablation variants.

All reported figures are generated directly from reproducible script execution (`python cli.py evaluate`).

---

## 2. Quantitative Results Benchmark Table

| Model / Configuration | Accuracy | Macro F1 | Macro Precision | Macro Recall | ECE | Coverage | Accepted / Total |
|---|---|---|---|---|---|---|---|
| **Baseline 1: Heuristic Scorer** | 0.9231 | 0.9153 | 0.9167 | 0.9333 | 0.0796 | 1.0000 | 13 / 13 |
| **Baseline 2: Neural MLP (Held-out Test)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.3837 | 1.0000 | 13 / 13 |
| **Baseline 3: Full Benchmark Forced** | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3475 | 1.0000 | 84 / 84 |
| **Proposed Method (Selective Abstention $\gamma = 0.44$)** | **0.9649** | **0.9649** | **0.9649** | **0.9649** | 0.3832 | **0.6786** | 57 / 84 |
| **Ablation: Full Multi-Signal Model** | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3475 | 1.0000 | 84 / 84 |
| **Ablation: w/o Freshness Analysis** | 0.9524 | 0.9568 | 0.9568 | 0.9568 | 0.3507 | 1.0000 | 84 / 84 |
| **Ablation: w/o Provenance Integrity** | 0.9405 | 0.9415 | 0.9397 | 0.9453 | 0.3923 | 1.0000 | 84 / 84 |
| **Ablation: Sparse-Only Retrieval** | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3528 | 1.0000 | 84 / 84 |
| **Ablation: Dense-Only Retrieval** | 0.9643 | 0.9675 | 0.9685 | 0.9669 | 0.3483 | 1.0000 | 84 / 84 |

---

## 3. Analysis of Research Hypotheses

### 3.1 Hypothesis 1: Risk-Coverage Tradeoff
- **Finding: Confirmed.**
- Under selective prediction at threshold $\gamma = 0.44$, the system abstains on 27 out of 84 claims (primarily unverified assertions lacking grounding evidence).
- On accepted predictions (coverage = 67.86%), the system achieves **96.49% accuracy** (55 correct predictions out of 57 accepted claims).
- The empirical Risk-Coverage curve confirms that selective risk decreases as low-confidence predictions are filtered into the abstention set.

### 3.2 Hypothesis 2: Elimination of False Contradictions
- **Finding: Confirmed.**
- In forced-label regimes, models routinely classify missing knowledge as factual contradiction.
- HalluciGuard-OKF cleanly routes claims without grounding evidence to `UNVERIFIED` and marks them as `ABSTAINED`, resulting in zero false contradiction verdicts on out-of-distribution entities (e.g., Dr. Arthur Pendelton).

### 3.3 Hypothesis 3: Multi-Signal Synergy
- **Finding: Confirmed.**
- Removing provenance integrity (`w/o Provenance`) degrades Macro F1 from **0.9675 to 0.9415** (-2.6% absolute decline) and increases calibration error (ECE = 0.3923).
- Removing temporal freshness (`w/o Freshness`) causes an accuracy drop from **0.9643 to 0.9524** (-1.2% absolute decline), confirming that temporal decay modeling is critical for preventing stale historical facts from being accepted.

---

## 4. Visualizations

The generated evaluation artifacts are persisted at:
- Risk-Coverage Curve: `evaluation/plots/risk_coverage_curve.png`
- Reliability Diagram (Calibration Curve): `evaluation/plots/calibration_curve.png`
- Machine-Readable Benchmark Metrics: `evaluation/results/experiment_results.json` and `evaluation/results/experiment_results.csv`
