# SECURITY AND VULNERABILITY AUDIT: HALLUCIGUARD-OKF

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Target:** Application Security, LLM-Specific Vulnerabilities (OWASP Top 10 for LLM), and AI Data Integrity  
**Auditor Roles:** Application Security Engineer, Data Privacy and Responsible AI Specialist, Independent Technical Auditor  
**Date:** November 3, 2026  
**Status:** COMPLETE  

---

## 1. Executive Security Assessment

The security posture of the **HalluciGuard-OKF** codebase reflects an early development environment without production hardening. While the repository contains no hardcoded API secrets or leaked credentials, it exhibits **two Critical vulnerabilities**:
1. **Arbitrary Code Execution via Unsafe Deserialization:** The use of `torch.load()` without `weights_only=True` and `joblib.load()` permits arbitrary code execution if model artifacts or scalers are tampered with.
2. **Denial of Service via Catastrophic Regex Backtracking (ReDoS):** Complex unbounded regular expressions in the claim extraction and NLI engines can hang CPU threads when processing crafted inputs.

Furthermore, while the documentation promises JWT authentication, role-based access control (RBAC), and audited API endpoints, **zero authentication or authorization controls currently exist in code (`NOT IMPLEMENTED`)**.

---

## 2. Dedicated Vulnerability Register

---

### VULN-01: Arbitrary Code Execution via Unsafe PyTorch & Joblib Deserialization
- **Vulnerability Description:**
  In [inference.py#L76](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L76) and [ml/scoring/models.py#L148](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/models.py#L148), serialized model weights and feature scalers are loaded from disk using standard `joblib.load()` and `torch.load()`. In PyTorch prior to default `weights_only=True` enforcement and in standard Python `joblib`/`pickle`, deserialization allows arbitrary Python code execution via `__reduce__` exploit payloads.
- **Attack Scenario:**
  An attacker with write access to the filesystem, a CI/CD build cache, or an external model registry substitutes `best_model.pt` or `feature_scaler.pkl` with a weaponized pickle payload containing a reverse shell or file exfiltration script. When `HalluciGuardPipeline()` initializes, the payload executes with the privileges of the running application process.
- **Preconditions:** Ability to write or replace files in `models/multisignal_head/`.
- **Potential Impact:** Full host compromise, credential theft, lateral movement.
- **Severity:** **Critical (CVSS 9.8)**
- **Evidence:**
  - [inference.py#L76](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/inference.py#L76): `self.scaler = joblib.load(scaler_path)`
  - [ml/scoring/models.py#L148](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/scoring/models.py#L148): `state_dict = torch.load(path, map_location=self.device)`
- **Recommended Mitigation:**
  1. Enforce `torch.load(path, map_location=self.device, weights_only=True)`.
  2. Transition model serialization to Hugging Face `safetensors` format.
  3. Replace `joblib.load()` with JSON-serialized scaler parameters (mean, variance, scale arrays).
- **Verification Test:**
  Create a test script loading a dummy pickle file containing `os.system('echo EXPLOITED')`. Verify that `weights_only=True` rejects non-tensor objects with `pickle.UnpicklingError`.

---

### VULN-02: Regular Expression Denial of Service (ReDoS) in Claim Extraction
- **Vulnerability Description:**
  In [ml/claim_extraction/extractor.py#L87-L90](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L87-L90), the regex pattern:
  `r"^(?P<subj>.+?\b(?:was|were|is|are)?\s*\w+ed)\s+in\s+(?P<year>1[789]\d{2}|20\d{2})\s+by\s+(?P<agent>[A-Z][a-zA-Z\s]+)[.]?$"`
  and line 164:
  `r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"`
  contain nested quantifiers and overlapping sub-patterns (`.+?\b`, `\s*`, `\w+ed`, `[a-zA-Z\s]+`).
- **Attack Scenario:**
  An attacker submits a large response payload composed of repetitive capitalized tokens or non-matching phrases ending without a terminal anchor (e.g., `"A " * 5000 + "ed in 2008 by " + "B " * 5000 + "!"`). The regex engine experiences polynomial or exponential backtracking, consuming 100% of a CPU core and blocking inference for other users.
- **Preconditions:** Ability to send arbitrary text to the inference pipeline or CLI.
- **Potential Impact:** Denial of service, CPU exhaustion, worker starvation.
- **Severity:** **High (CVSS 7.5)**
- **Evidence:**
  - [ml/claim_extraction/extractor.py#L87-L90](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L87-L90)
  - [ml/claim_extraction/extractor.py#L164](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/ml/claim_extraction/extractor.py#L164)
- **Recommended Mitigation:**
  1. Replace custom regexes with deterministic linear-time NLP tokenizers (e.g. spaCy or Python `re2` engine).
  2. Implement an input string length ceiling (e.g., maximum 4,096 characters per response).
- **Verification Test:**
  Send a 10,000-character repetitive string to `ClaimExtractor.extract_claims()` and benchmark execution duration (must terminate in $< 50$ ms).

---

### VULN-03: Complete Absence of Authentication & Authorization Controls
- **Vulnerability Description:**
  [README.md#L1491-L1506](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/README.md#L1491-L1506) documents JWT authentication and an RBAC model (`admin`, `researcher`, `analyst`, `viewer`). In reality, no authentication checks, token decoders, or permission barriers exist anywhere in the codebase (`NOT IMPLEMENTED`).
- **Attack Scenario:**
  If the system were wrapped in a naive HTTP handler in its current state, any network actor could submit arbitrary inference requests, trigger computationally expensive verifications, or query internal knowledge artifacts without credentials.
- **Preconditions:** Exposing the pipeline via network socket.
- **Potential Impact:** Unauthorized access, service abuse, data scraping.
- **Severity:** **High (CVSS 7.5)**
- **Evidence:**
  - Directory `backend/` and files `auth.py` are completely absent from the filesystem.
- **Recommended Mitigation:**
  Implement OAuth2 / JWT bearer authentication middleware in FastAPI using `pyjwt` or `auth0-python`, validating scopes on all ingestion and verification endpoints.
- **Verification Test:**
  Issue an unauthenticated request to the API; confirm HTTP 401 Unauthorized response.

---

### VULN-04: Lack of Cryptographic Integrity on Knowledge Artifacts
- **Vulnerability Description:**
  In [knowledge/store.py#L23-L37](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/store.py#L23-L37), OKF artifacts are ingested directly from JSON files. Although the Pydantic validator checks structural typing, it does not verify digital signatures, checksums, or source authenticity.
- **Attack Scenario:**
  An adversary modifies `knowledge/artifacts/curated_knowledge.json` to insert a fraudulent claim asserting that a phishing domain is an official institutional resource, marking it `"verification_status": "verified"` with `"reliability_estimate": 1.0`. The system blindly trusts this artifact during retrieval, resulting in false `SUPPORTED` classifications for malicious URLs or false factual claims.
- **Preconditions:** Local or network write capability to knowledge data files.
- **Potential Impact:** Knowledge base poisoning, hallucination detector spoofing.
- **Severity:** **Medium (CVSS 6.5)**
- **Evidence:**
  - [knowledge/store.py#L32-L36](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/knowledge/store.py#L32-L36)
  - [okf/validator/validator.py#L33-L60](file:///c:/Users/ishar/Desktop/Hallucination_detection_system/okf/validator/validator.py#L33-L60)
- **Recommended Mitigation:**
  Add an `integrity` schema block to OKF containing an HMAC or Ed25519 digital signature generated by an authorized curation key. The store must reject unsigned or invalid artifacts.
- **Verification Test:**
  Attempt to load a modified artifact with an altered statement; verify rejection due to signature mismatch.

---

### VULN-05: Missing Dependency Manifest and Pinned Versions
- **Vulnerability Description:**
  The repository does not supply a `requirements.txt`, `Pipfile`, or `pyproject.toml`.
- **Attack Scenario:**
  Developers or deployment runners execute `pip install torch transformers scikit-learn` without version pinning. A malicious typo-squatted package or a compromised upstream release installs backdoored code into the execution environment.
- **Preconditions:** Standard deployment setup.
- **Potential Impact:** Supply-chain compromise, build non-reproducibility.
- **Severity:** **Medium (CVSS 5.5)**
- **Evidence:** Root directory contains only `inference.py`, `train.py`, and `README.md`.
- **Recommended Mitigation:**
  Generate a strict, hash-pinned `pyproject.toml` and lockfile using `uv` or `pip-tools`.
- **Verification Test:**
  Execute `uv sync --frozen` to confirm deterministic environment reconstruction.

---

## 3. OWASP Top 10 for LLM Applications Compliance Matrix

| OWASP LLM Category | Description | HalluciGuard-OKF Audit Finding | Status |
| :--- | :--- | :--- | :--- |
| **LLM01: Prompt Injection** | Manipulating model via untrusted inputs | Low risk in current state because LLMs are not used as judges; text is processed via regex/MLP. However, injected text reflects into explanation strings. | Partially Mitigated |
| **LLM02: Sensitive Info Disclosure** | Leaking confidential data in outputs | System outputs full internal evidence metadata and source filenames in explanation payloads. | Medium Risk |
| **LLM03: Supply Chain Vulnerabilities** | Insecure libraries or models | Unpinned dependencies and pickled weights create severe supply chain exposure. | **High Risk** |
| **LLM04: Data and Model Poisoning** | Tampering with training/retrieval data | In-memory knowledge store lacks cryptographic signatures; benchmark data can be easily poisoned. | **High Risk** |
| **LLM05: Improper Output Handling** | Unsanitized output rendered in downstream apps | `_build_explanation()` injects raw claim and evidence strings into JSON without HTML/markdown escaping. | Medium Risk |
| **LLM06: Excessive Agency** | Granting autonomous action capabilities | System is read-only analysis pipeline; does not invoke external actions. | No Risk |
| **LLM07: System Prompt Leakage** | Exposing internal prompts | No system prompts are used in the current rule/MLP engine. | No Risk |
| **LLM08: Vector & Embedding Weaknesses** | Exploiting retrieval similarity | Fallback TF-IDF vectorizer can be easily gamed via keyword stuffing to manipulate retrieval scores. | **High Risk** |
| **LLM09: Misinformation & Overreliance** | User over-trusts false predictions | High confidence scores ($0.95$) returned on brittle regex matches encourage false user trust. | **High Risk** |
| **LLM10: Model Denial of Service** | Resource exhaustion via expensive inputs | Vulnerable to ReDoS in regex parsing; lack of input length caps allows memory exhaustion. | **High Risk** |
