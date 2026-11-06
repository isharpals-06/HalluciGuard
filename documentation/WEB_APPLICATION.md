# HalluciGuard-OKF Web Application Guide

## 1. Overview

HalluciGuard-OKF features an interactive single-page web dashboard built on FastAPI and modern responsive HTML5/CSS3/Vanilla JavaScript. It allows researchers and evaluators to observe the claim extraction, hybrid retrieval, multi-signal feature scoring, and uncertainty-aware selective abstention behavior in real-time.

---

## 2. Launching the Web Server

Launch the application using Uvicorn:

```bash
# Using active virtual environment
.venv\Scripts\python.exe -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Once running, navigate to:
```
http://localhost:8000
```

---

## 3. User Interface Features

### 3.1 Live Verification Dashboard
- **Input Text Area:** Supports arbitrary unstructured multi-sentence text.
- **Quick Preset Selectors:**
  - *Speed of Light:* Demonstrates a numerical claim verified by NIST evidence (`SUPPORTED`).
  - *Sydney Capital:* Demonstrates a factual geographic contradiction caught by Geoscience Australia evidence (`CONTRADICTED`).
  - *Arthur Pendelton:* Demonstrates an ungrounded fictitious assertion that triggers safe abstention rather than false contradiction (`UNVERIFIED` / `ABSTAINED`).
  - *Multi-Claim:* Demonstrates composite paragraphs containing both valid and hallucinated propositions.
- **Selective Abstention Slider:** Real-time threshold slider from 0.00 to 0.80. Adjusting the slider demonstrates how low-confidence predictions cleanly transition into the abstention set.
- **Provider Selector:** Switch between local hybrid verification and cloud model integrations (Gemini, OpenAI).

### 3.2 Dynamic Results Breakdown
- **Overall Status Banner:** Color-coded status header (`VERIFIED` in emerald, `HALLUCINATED` in crimson, `UNVERIFIED` in amber).
- **Claim Card Drill-down:**
  - Syntactic claim type badge (`numerical`, `entity`, `temporal`, `factual`).
  - Decision status badge (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`, `ABSTAINED`).
  - Classifier confidence progress bar (calibration remains preliminary).
  - Uncertainty score progress bar (heuristic proxy, not validated epistemic uncertainty).
  - Evidence Grounding card showing the exact OKF object ID, primary source name, reliability score, and statement snippet.
  - Natural language explanation summarizing why the claim was accepted, contradicted, or abstained.

### 3.3 Benchmark & Ablation Tab
- Interactive metrics table populated dynamically via `GET /api/benchmark`.
- Embedded high-resolution plots:
  - Risk-Coverage curve.
  - Reliability diagram (Calibration curve).

### 3.4 Knowledge Base Browser
- Searchable catalog of all 31 verified OKF artifacts, detailing entity names, statements, source URLs, freshness states, and reliability estimates.

### 3.5 System Architecture View
- Visual schematic detailing the 6-stage verification flow from atomic parsing to selective prediction.
