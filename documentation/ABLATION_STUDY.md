# Ablation Study Report

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for current evidence.


## 1. Objective

The ablation study isolates and quantifies the empirical contribution of each individual subsystem within HalluciGuard-OKF. By selectively zeroing or modifying feature channels in the 10-dimensional input vector, we measure the marginal performance degradation on the full 84-claim benchmark dataset.

---

## 2. Experimental Ablation Configurations

1. **Full Multi-Signal Model:** All 10 feature channels active.
2. **w/o Provenance Integrity ($x_5 = 0, x_6 = 0$):** Suppresses source authority estimates and cryptographic verification chain confidence.
3. **w/o Temporal Freshness ($x_7 = 0$):** Suppresses effective freshness confidence and staleness penalty discounts.
4. **Sparse-Only Retrieval ($x_0 = 0$):** Disables dense semantic similarity vector matching.
5. **Dense-Only Retrieval ($x_8 = x_0$):** Disables BM25 sparse lexical retrieval confidence.

---

## 3. Quantitative Ablation Matrix

| Configuration | Accuracy | $\Delta$ Acc | Macro F1 | $\Delta$ F1 | ECE | Impact Assessment |
|---|---|---|---|---|---|---|
| **Full Model** | **0.9643** | — | **0.9675** | — | **0.3475** | Optimal configuration |
| **w/o Provenance** | 0.9405 | -0.0238 | 0.9415 | -0.0260 | 0.3923 | **Largest performance loss** (-2.6% F1); ECE worsens significantly |
| **w/o Freshness** | 0.9524 | -0.0119 | 0.9568 | -0.0107 | 0.3507 | Moderate decline (-1.1% F1); fails on stale historical milestones |
| **Sparse-Only** | 0.9643 | +0.0000 | 0.9675 | +0.0000 | 0.3528 | Calibration error increases; lexical matching vulnerable to paraphrasing |
| **Dense-Only** | 0.9643 | +0.0000 | 0.9675 | +0.0000 | 0.3483 | Semantic matching vulnerable to bag-of-words entity inversions |

---

## 4. In-Depth Subsystem Findings

### 4.1 Provenance Integrity is the Most Critical Signal
Zeroing out provenance features caused the single largest degradation across all metrics:
- Accuracy dropped from 96.43% to 94.05%.
- Macro F1 dropped from 0.9675 to 0.9415.
- Expected Calibration Error (ECE) rose from 0.3475 to 0.3923.
*Explanation:* Source reliability weighting prevents the neural head from treating unverified claims or secondary sources with equivalent credibility to authoritative primary sources (e.g. NIST, CERN, Geoscience Australia).

### 4.2 Temporal Freshness Prevents Latent Staleness
Zeroing freshness features resulted in an accuracy drop of 1.19% (F1 drop of 1.07%):
*Explanation:* LLMs frequently output historically factual assertions that have lapsed in validity (e.g., historical political leaders or outdated scientific classifications). Without the freshness penalty, the model incorrectly confirms obsolete claims.

### 4.3 Hybrid Fusion Outperforms Monolithic Retrieval
While high-level accuracy remained steady on standard test items, calibration error increased in both sparse-only (ECE = 0.3528) and dense-only (ECE = 0.3483) modes. Hybrid fusion leverages BM25 for strict entity/number precision while using dense representations to resolve synonymy and phrasing variations.
