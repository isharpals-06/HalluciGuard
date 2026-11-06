import numpy as np
import pytest

from evaluation.calibration import apply_temperature, fit_temperature, select_abstention_threshold
from evaluation.metrics import evaluate_predictions
from evaluation.perturbations import perturb
from ml.nli.nli_engine import NLIEngine
from datasets.adapters import adapt_triviaplus_rows, load_ragtruth
from datasets.loader import BenchmarkDataLoader


def test_nli_mode_is_explicit_and_rule_based_is_named():
    engine = NLIEngine(mode="rule_based")
    assert engine.mode == "rule_based"
    with pytest.raises(ValueError):
        NLIEngine(mode="automatic")


def test_transformer_mode_does_not_silently_fall_back(monkeypatch):
    def unavailable(self):
        raise RuntimeError("weights unavailable")
    monkeypatch.setattr(NLIEngine, "_load_transformer", unavailable)
    with pytest.raises(RuntimeError, match="weights unavailable"):
        NLIEngine(model_name="example/nli", mode="transformer")


def test_conflict_condition_is_distinct_and_forces_insufficient_status():
    example = {
        "example_id": "x1",
        "claim_text": "A claim",
        "claim_status": "SUPPORTED",
        "evidence_items": [
            {"evidence_id": "s1", "text": "support", "relation": "SUPPORT"},
        ],
    }
    changed = perturb(
        example,
        "contradiction_insertion",
        seed=7,
        distractors=[{"evidence_id": "c1", "text": "contradiction", "relation": "CONTRADICT"}],
    )
    assert changed["claim_status"] == "INSUFFICIENT_EVIDENCE"
    assert changed["evidence_condition"] == "CONFLICTING_EVIDENCE"
    assert changed["label_validation_status"] == "requires_human_validation"
    assert changed["original_example_id"] == "x1"


def test_order_permutation_is_deterministic_and_preserves_state():
    example = {
        "example_id": "x2",
        "claim_text": "A claim",
        "evidence_items": [
            {"evidence_id": f"s{i}", "text": f"evidence {i}", "relation": "SUPPORT"}
            for i in range(8)
        ],
    }
    first = perturb(example, "order_permutation", seed=9)
    second = perturb(example, "order_permutation", seed=9)
    assert first["transformed_evidence_ids"] == second["transformed_evidence_ids"]
    assert first["claim_status"] == "SUPPORTED"
    assert first["evidence_condition"] == "SUPPORT_ONLY"


def test_calibration_and_metrics_cover_brier_nll_and_pr_auc():
    labels = np.array([0, 1, 2, 0, 1, 2])
    logits = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]] * 2, dtype=float)
    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    probs = apply_temperature(logits, temperature)
    metrics = evaluate_predictions(labels, probs.argmax(axis=1), probs)
    assert {"brier_score", "negative_log_likelihood", "macro_pr_auc"} <= metrics.keys()
    assert metrics["accuracy"] == 1.0


def test_abstention_threshold_is_selected_from_validation_data():
    probs = np.array([[0.95, 0.03, 0.02], [0.10, 0.80, 0.10], [0.55, 0.35, 0.10]])
    threshold, operating_point = select_abstention_threshold(
        probs, np.array([0, 1, 2]), target_risk=0.0
    )
    assert threshold == pytest.approx(0.80)
    assert operating_point["accepted_count"] == 2
    assert operating_point["risk"] == 0.0


def test_training_loader_uses_retrieval_features_and_scenario_disjoint_splits():
    data = BenchmarkDataLoader().load_and_preprocess()
    split_groups = data["split_group_ids"]
    assert data["evidence_mode"] == "retrieval"
    assert data["splits"]["train"][0].shape[1] == 15
    assert not (set(split_groups["train"]) & set(split_groups["validation"]))
    assert not (set(split_groups["train"]) & set(split_groups["test"]))
    assert not (set(split_groups["validation"]) & set(split_groups["test"]))


def test_triviaplus_adapter_preserves_supplementary_and_split():
    rows = adapt_triviaplus_rows([{
        "id": "t1", "split": "test", "question": "q", "answer": "a", "article": "c",
        "answer_sentence_list": ["supported", "not in context", "extra"],
        "sentence_level_majority_vote": ["supports", "not-mentioned", "supplementary"],
    }])
    claims = rows[0]["claims"]
    assert [claim["claim_status"] for claim in claims] == [
        "SUPPORTED", "INSUFFICIENT_EVIDENCE", "SUPPLEMENTARY"
    ]
    assert all(claim["split"] == "test" for claim in claims)
    with pytest.raises(ValueError):
        adapt_triviaplus_rows([{"answer_sentence_list": ["a"], "sentence_level_majority_vote": []}])


def test_ragtruth_adapter_preserves_span_annotations(tmp_path):
    response_path = tmp_path / "responses.jsonl"
    source_path = tmp_path / "sources.jsonl"
    response_path.write_text(
        '{"id":"r1","source_id":"s1","split":"test","response":"hello","labels":[{"start":0,"end":5,"text":"hello"}]}\n',
        encoding="utf-8",
    )
    source_path.write_text(
        '{"source_id":"s1","task_type":"QA","source_info":{"question":"q","passages":"p"}}\n',
        encoding="utf-8",
    )
    row = load_ragtruth(response_path, source_path)[0]
    assert row["hallucination_spans"][0]["text"] == "hello"
    assert row["claim_alignment"] == "not_performed"
