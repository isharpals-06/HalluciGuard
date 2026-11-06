# TESTING AND VALIDATION PLAN: HALLUCIGUARD-OKF

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](../RESEARCH_REPORT.md) for current evidence.


**Audit Target:** Quality Assurance Strategy, Automated Test Protocols, and Benchmark Validation  
**Auditor Roles:** AI Evaluation and Benchmarking Specialist, NLP Research Engineer, Senior Software Architect  
**Date:** November 3, 2026  
**Status:** COMPLETE  

---

## 1. Quality Assurance Strategy Overview

The testing framework must transition from validating hardcoded toy examples to verifying generalized, robust hallucination detection across four testing layers:
1. **Unit Testing Layer:** Validates individual component isolation (Regex parsers, Pydantic schemas, normalization scalers).
2. **Integration Testing Layer:** Validates end-to-end dataflow through `HalluciGuardPipeline` across mock and real model instances.
3. **Adversarial & Stress Testing Layer:** Systematically evaluates the 14 adversarial failure modes identified in Phase 4.
4. **Statistical Benchmarking Layer:** Executes regression testing against external standardized corpora (FEVER, FActScore, RAGTruth).

---

## 2. Unit & Integration Test Specifications

### 2.1 Atomic Claim Extraction Test Suite (`tests/unit/test_claim_extraction_robust.py`)
- **TC-EXT-01 (Complex Compound Sentences):**
  - Input: *"Tesla, which was founded in 2003, produced its first Roadster in 2008 and went public in 2010."*
  - Expected: 3 claims extracted with correct temporal references.
- **TC-EXT-02 (Passive Voice without Explicit Agent):**
  - Input: *"The discovery was confirmed in 1998 by European astronomers."*
  - Expected: Decomposes into discovery claim and attribution claim.
- **TC-EXT-03 (ReDoS Stress Test):**
  - Input: 5,000 repetitive non-terminal tokens: `"Alpha " * 1000 + "was founded in 2008 by Beta."`
  - Expected: Completes in $< 50$ ms without infinite recursion.

### 2.2 Natural Language Inference Test Suite (`tests/unit/test_nli_semantic.py`)
- **TC-NLI-01 (Paraphrase Invariance):**
  - Premise: *"Tesla was established by Martin Eberhard and Marc Tarpenning."*
  - Hypothesis: *"The company's founding was initiated by Tarpenning and Eberhard."*
  - Expected: $P(\text{entailment}) \ge 0.85$, $P(\text{contradiction}) \le 0.05$.
- **TC-NLI-02 (Double Negation Resolution):**
  - Premise: *"Neil Armstrong stepped onto the lunar surface in 1969."*
  - Hypothesis: *"It is not false that Armstrong walked on the moon in 1969."*
  - Expected: $P(\text{entailment}) \ge 0.80$, $P(\text{contradiction}) \le 0.10$.
- **TC-NLI-03 (Numerical Range Consistency):**
  - Premise: *"The speed of light is 299,792,458 m/s."*
  - Hypothesis: *"Light travels at approximately 3.0 x 10^8 meters per second."*
  - Expected: $P(\text{entailment}) \ge 0.80$.

---

## 3. Automated Adversarial Test Harness

The 14 scenarios analyzed in the Failure Analysis audit must be codified into an automated regression test script: `tests/adversarial/test_adversarial_suite.py`.

```python
import pytest
from inference import HalluciGuardPipeline

@pytest.fixture(scope="module")
def pipeline():
    return HalluciGuardPipeline()

ADVERSARIAL_CASES = [
    # Case 1: Plausible falsehood
    ("What did Einstein win Nobel Prize for?", 
     "Einstein won the 1921 Nobel Prize in Physics for General Relativity.", 
     "CONTRADICTED"),
    
    # Case 8: Negation manipulation
    ("Did Apollo 11 land on the moon?", 
     "It is not untrue that Apollo 11 landed on the moon on July 20, 1969.", 
     "VERIFIED"),
    
    # Case 14: Adversarial surface tampering
    ("Where is Australia's capital?", 
     "The capital city of Australia is Sydney, Canberra.", 
     "HALLUCINATED"),
]

@pytest.mark.parametrize("query, response, expected_status", ADVERSARIAL_CASES)
def test_adversarial_scenarios(pipeline, query, response, expected_status):
    result = pipeline.analyze(response_text=response, query=query)
    assert result["overall_status"] == expected_status, (
        f"Failed on adversarial case: {query} -> {response}. "
        f"Expected {expected_status}, got {result['overall_status']}"
    )
```

---

## 4. Benchmark Validation & Continuous Regression Protocol

To guarantee scientific reproducibility and prevent regression during future development, the CI/CD pipeline must enforce the following validation gates:

```
┌─────────────────────────────────────────────────────────────┐
│                 Continuous Validation Gate                  │
├─────────────────────────────────────────────────────────────┤
│ 1. Code Security Scan (Bandit / Trivy):                     │
│    - Zero High or Critical security findings                │
│    - Zero unverified pickle/joblib deserialization calls    │
│                                                             │
│ 2. Unit & Integration Test Gate (Pytest):                   │
│    - 100% test pass rate across unit suite                  │
│    - Code test coverage >= 85%                              │
│                                                             │
│ 3. Benchmark Regression Thresholds:                         │
│    - FEVER Held-out Split (N=1,000): Macro F1 >= 0.78       │
│    - FActScore Extraction Accuracy: >= 82%                  │
│    - Expected Calibration Error (ECE): <= 0.12              │
│                                                             │
│ 4. Performance Latency SLA:                                 │
│    - End-to-end single-sentence verification: <= 80 ms (CPU)│
│    - Qdrant Vector Retrieval Latency: <= 12 ms (p95)        │
└─────────────────────────────────────────────────────────────┘
```
