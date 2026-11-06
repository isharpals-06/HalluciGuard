# Threat Model: HalluciGuard-OKF

## 1. Threat Modeling Overview

This document formalizes the threat landscape for HalluciGuard-OKF using the STRIDE threat categorization model:
- **S**poofing
- **T**ampering
- **R**epudiation
- **I**nformation Disclosure
- **D**enial of Service
- **E**levation of Privilege

---

## 2. STRIDE Analysis Matrix

| Threat Category | Threat Description | Attack Vector | Severity | Implemented Mitigation |
|---|---|---|---|---|
| **Spoofing** | Attacker impersonates a trusted primary source or injects counterfeit OKF artifacts. | Fabricating JSON artifacts claiming NIST or CERN provenance. | High | Cryptographic signature validation on OKF schemas; `ProvenanceAnalyzer` enforces verified primary source domain whitelisting. |
| **Tampering** | Attacker modifies weights or feature scaler to bias verification toward approving hallucinations. | Modifying model checkpoint file on disk. | High | SafeTensors format eliminates arbitrary code execution; file checksums and read-only container volumes prevent runtime tampering. |
| **Repudiation** | An actor denies having generated or approved a hallucinated response. | Inability to audit verification history. | Medium | Unique immutable `analysis_id` and `request_id` generated per verification run; timestamped evidence citations logged. |
| **Information Disclosure** | Leakage of proprietary knowledge base artifacts or API keys. | API error tracebacks or improper logging. | Medium | FastAPI exception handlers catch errors cleanly without dumping internal stack traces; API keys read via environment variables. |
| **Denial of Service** | Attacker submits pathological text strings to exhaust CPU resources. | ReDoS regex attack with repeating nested patterns. | High | Input length cap (4,096 characters); bounded non-backtracking regex expressions; verified $<50$ ms completion. |
| **Elevation of Privilege** | Attacker achieves arbitrary code execution via model loading. | Deserializing malicious pickled model payload. | Critical | Zero pickle dependency: weights loaded via `safetensors.torch`; scaler loaded via `SafeFeatureScaler` (pure JSON). |

---

## 3. Adversarial Robustness Assessment

### 3.1 Prompt Injection Isolation
- **Mechanism:** Text is segmented into non-executable strings and mapped into continuous embeddings and signal vectors. Instruction-like commands (`"Ignore previous rules..."`) are treated purely as passive propositional statements, neutralized, and evaluated against the knowledge store.

### 3.2 Epistemic Uncertainty Under Distribution Shift
- **Mechanism:** When inputs diverge from the training distribution or lack supporting evidence in the knowledge store, epistemic uncertainty rises automatically towards 1.0, triggering selective abstention rather than producing high-confidence false classifications.
