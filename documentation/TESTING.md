# Testing and Quality Assurance Documentation

## Current verification run (2026-11-05)

Command: `.venv\Scripts\python.exe -m pytest tests -q`  
Environment: Python 3.11.15, pytest 9.2.0, Windows, CPU.  
Outcome: **38 passed, 0 failed, 0 skipped in 21.05 seconds.** This run includes 9 research-validity tests for explicit mode selection, grouped splits, local dataset adapters, calibration metrics, selective threshold selection, and perturbation lineage. It does not validate external datasets, transformer weights, or a full perturbation experiment.

> Historical audit snapshot (pre-2026-11-05 implementation). Metrics and verification claims below describe an earlier code/artifact state and are not current results. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for current evidence.


## 1. Testing Framework and Architecture

HalluciGuard-OKF features a three-tiered test architecture implemented with `pytest`:
1. **Unit Tests (`tests/unit/`):** Component-level tests verifying atomic claim extraction, OKF schema validation, BM25 retrieval, NLI heuristic rules, provenance analysis, and freshness decay.
2. **Adversarial & Robustness Tests (`tests/adversarial/`):** Tests verifying ReDoS resistance, SafeTensors deserialization, prompt injection neutralization, double negation handling, and selective abstention.
3. **Integration Tests (`tests/integration/`):** End-to-end tests verifying CLI command execution and FastAPI REST endpoints.

---

## 2. Test Execution

Execute the entire test suite using the virtual environment:

```bash
# Run all tests with summary
pytest tests -v

# Run only unit tests
pytest tests/unit -v

# Run only adversarial robustness tests
pytest tests/adversarial -v

# Run only integration tests
pytest tests/integration -v
```

---

## 3. Inventory of Implemented Tests

### 3.1 Unit Tests (`tests/unit/`, 13 Tests)
1. `test_claim_extraction.py::test_atomic_decomposition` — Validates sentence clause splitting.
2. `test_claim_extraction.py::test_multi_sentence_extraction` — Validates extraction across paragraphs.
3. `test_claim_extraction.py::test_empty_input` — Validates blank input handling.
4. `test_end_to_end_pipeline.py::test_supported_claim_pipeline` — Verifies speed of light pipeline execution.
5. `test_end_to_end_pipeline.py::test_contradicted_claim_pipeline` — Verifies Australia/Sydney contradiction detection.
6. `test_end_to_end_pipeline.py::test_unverified_claim_pipeline` — Verifies ungrounded claim routing to UNVERIFIED.
7. `test_okf.py::test_valid_okf_creation_and_validation` — Tests Pydantic v2 OKF schema compliance.
8. `test_okf.py::test_invalid_okf_rejected` — Tests rejection of malformed OKF objects.
9. `test_okf.py::test_temporal_consistency` — Tests temporal validity window constraints.
10. `test_retrieval_and_verification.py::test_bm25_retrieval` — Validates Okapi BM25 sparse retrieval.
11. `test_retrieval_and_verification.py::test_nli_date_contradiction` — Validates temporal clash detection.
12. `test_retrieval_and_verification.py::test_provenance_analyzer` — Validates primary source reliability scoring.
13. `test_retrieval_and_verification.py::test_freshness_analyzer` — Validates STALE status assignment.

### 3.2 Adversarial & Security Tests (`tests/adversarial/`, 8 Tests)
1. `test_redos_protection` — Executes 20,000-char repeating regex attack payload; verifies completion $<200$ ms.
2. `test_safe_scaler_deserialization` — Verifies JSON scaler loads without pickle.
3. `test_prompt_injection_isolation` — Verifies system override prompts are isolated as claims.
4. `test_missing_evidence_not_contradicted` — Verifies missing evidence does NOT cause false contradiction.
5. `test_double_negation_resolution` — Verifies double negation (`not untrue`) resolves to entailment.
6. `test_attribute_clash_entity_grounding` — Verifies incorrect attribute for known entity yields CONTRADICTED.
7. `test_empty_and_whitespace_inputs` — Verifies clean error response on whitespace/empty text.
8. `test_selective_abstention_behavior` — Verifies abstention trigger under low confidence and absent evidence.

### 3.3 Integration Tests (`tests/integration/`, 8 Tests)
1. `test_cli_info_command` — Verifies `python cli.py info` exit code 0 and telemetry output.
2. `test_cli_verify_supported` — Verifies `python cli.py verify` with verified claim returns 0.
3. `test_cli_verify_hallucination` — Verifies `python cli.py verify` with contradicted claim returns non-zero exit code.
4. `test_web_api_info_endpoint` — Tests `GET /api/info` returns HTTP 200 and ready state.
5. `test_web_api_knowledge_endpoint` — Tests `GET /api/knowledge` returns artifact list.
6. `test_web_api_benchmark_endpoint` — Tests `GET /api/benchmark` returns benchmark results.
7. `test_web_api_verify_endpoint` — Tests `POST /api/verify` returns structured JSON response.
8. `test_web_api_plots_endpoint` — Tests `GET /api/plots/risk_coverage` returns valid PNG bytes.

---

## 4. Test Results Summary

- **Total Tests Collected:** 29
- **Total Tests Passed:** 29 (100% pass rate)
- **Total Tests Failed:** 0
- **Total Execution Time:** ~14.0 seconds
