# HalluciGuard-OKF REST API Reference

The HalluciGuard-OKF REST API is powered by FastAPI and provides endpoints for real-time text verification, benchmark introspection, knowledge base inspection, and system telemetry.

## Base URL
`http://localhost:8000`

---

## 1. Endpoints

### 1.1 Factual Verification
- **Route:** `POST /api/verify`
- **Description:** Decomposes input text into atomic claims, retrieves verified OKF evidence, computes multi-signal features, and returns evidence-grounded verification decisions.
- **Request Body (`application/json`):**
  ```json
  {
    "text": "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
    "query": "What is the speed of light in vacuum?",
    "abstain_threshold": 0.44,
    "provider": "local"
  }
  ```
- **Response (`application/json`):**
  ```json
  {
    "analysis_id": "76f20942-ef13-4449-8ddb-fbdb9ea06b79",
    "request_id": "REQ-263EE1D0B830",
    "overall_status": "VERIFIED",
    "hallucination_score": 0.05,
    "claims": [
      {
        "claim_id": "C001",
        "claim_text": "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
        "claim_type": "numerical",
        "status": "SUPPORTED",
        "confidence": 0.96,
        "uncertainty_score": 0.4125,
        "abstained": false,
        "abstention_reason": null,
        "flags": [],
        "evidence": [
          {
            "evidence_id": "EV-OKF-SCI-001",
            "okf_object_id": "OKF-SCI-001",
            "text": "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
            "relevance_score": 0.7773,
            "rank": 1
          }
        ],
        "provenance": {
          "source_name": "National Institute of Standards and Technology (NIST)",
          "source_type": "primary",
          "verified": true,
          "freshness_state": "CURRENT"
        },
        "explanation": "The claim is confirmed by verified evidence from National Institute of Standards and Technology (NIST)..."
      }
    ],
    "metrics": {
      "total_claims": 1,
      "supported_claims": 1,
      "contradicted_claims": 0,
      "unverified_claims": 0,
      "abstained_claims": 0,
      "coverage_rate": 1.0,
      "evidence_coverage": 1.0,
      "hallucination_rate": 0.0,
      "average_confidence": 0.96,
      "mean_uncertainty": 0.4125
    }
  }
  ```

---

### 1.2 System Telemetry
- **Route:** `GET /api/info`
- **Description:** Returns runtime environment, hardware acceleration status, SafeTensors support, and model loading state.
- **Response:**
  ```json
  {
    "status": "ready",
    "python_version": "3.11.15",
    "pytorch_version": "2.15.0+cpu",
    "cuda_available": false,
    "safetensors_support": "0.8.0",
    "verified_artifacts_count": 31,
    "model_loaded": true,
    "supported_providers": ["local", "gemini", "openai"]
  }
  ```

---

### 1.3 Knowledge Base Inspection
- **Route:** `GET /api/knowledge`
- **Description:** Returns the full catalog of verified OKF schema artifacts stored in the knowledge base.

---

### 1.4 Benchmark Telemetry
- **Route:** `GET /api/benchmark`
- **Description:** Returns the latest quantitative benchmark metrics for baselines, proposed selective prediction, and ablation experiments.

---

### 1.5 Research Plots
- **Route:** `GET /api/plots/{plot_name}`
- **Parameters:** `plot_name` $\in$ `{risk_coverage, calibration}`
- **Response:** `image/png` high-resolution visual figure.
