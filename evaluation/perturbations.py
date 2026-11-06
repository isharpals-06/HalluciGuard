"""Deterministic retrieval-context perturbations with explicit evidence-state updates."""

from copy import deepcopy
import random
from typing import Dict, List, Any

RELATIONS = {"SUPPORT", "CONTRADICT", "IRRELEVANT"}


def _validate(example: Dict[str, Any]) -> None:
    if not example.get("example_id") or not example.get("claim_text"):
        raise ValueError("A perturbation example requires example_id and claim_text.")
    evidence = example.get("evidence_items")
    if not isinstance(evidence, list):
        raise ValueError("evidence_items must be a list of evidence records.")
    for item in evidence:
        if not item.get("evidence_id") or not item.get("text"):
            raise ValueError("Every evidence record requires evidence_id and text.")
        if item.get("relation") not in RELATIONS:
            raise ValueError("Evidence relation must be SUPPORT, CONTRADICT, or IRRELEVANT.")


def _state(evidence: List[Dict[str, Any]]) -> tuple[str, str]:
    relevant = [item for item in evidence if item["relation"] != "IRRELEVANT"]
    support = any(item["relation"] == "SUPPORT" for item in relevant)
    contradict = any(item["relation"] == "CONTRADICT" for item in relevant)
    if support and contradict:
        return "INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"
    if support:
        return "SUPPORTED", "SUPPORT_ONLY"
    if contradict:
        return "CONTRADICTED", "CONTRADICTION_ONLY"
    return "INSUFFICIENT_EVIDENCE", "NO_RELEVANT_EVIDENCE" if not evidence else "IRRELEVANT_EVIDENCE"


def perturb(example: Dict[str, Any], transformation: str, *, seed: int,
            distractors: List[Dict[str, Any]] | None = None,
            position: str | None = None,
            evidence_id: str | None = None) -> Dict[str, Any]:
    """Transform evidence while retaining lineage and recalculating the evidence state.

    Contradictory insertions and support removals are marked `requires_human_validation`;
    the function does not assert that unreviewed annotations are ground truth.
    """
    _validate(example)
    result = deepcopy(example)
    original = deepcopy(example["evidence_items"])
    items = deepcopy(original)
    rng = random.Random(seed)
    validation = "validated_by_relation_metadata"
    params: Dict[str, Any] = {"seed": seed}

    if transformation == "order_permutation":
        rng.shuffle(items)
    elif transformation == "irrelevant_injection":
        additions = deepcopy(distractors or [])
        if not additions or any(item.get("relation") != "IRRELEVANT" for item in additions):
            raise ValueError("Distractor injection requires explicitly IRRELEVANT records.")
        existing = {item["evidence_id"] for item in items}
        if any(item.get("evidence_id") in existing for item in additions):
            raise ValueError("Injected evidence IDs must be unique.")
        items.extend(additions)
        rng.shuffle(items)
        params["distractor_ids"] = [item["evidence_id"] for item in additions]
    elif transformation == "position_change":
        if position not in {"beginning", "middle", "end"}:
            raise ValueError("position must be beginning, middle, or end")
        # Move the first relevant item to an explicit context position.
        idx = next((i for i, item in enumerate(items) if item["relation"] != "IRRELEVANT"), None)
        if idx is None:
            raise ValueError("Position change requires relevant evidence.")
        target = items.pop(idx)
        insert_at = {"beginning": 0, "middle": len(items) // 2, "end": len(items)}[position]
        items.insert(insert_at, target)
        params["position"] = position
    elif transformation == "support_removal":
        removed = [item for item in items if item["relation"] == "SUPPORT"]
        if evidence_id:
            removed = [item for item in removed if item["evidence_id"] == evidence_id]
        if not removed:
            raise ValueError("No matching supporting evidence to remove.")
        removed_ids = {item["evidence_id"] for item in removed}
        items = [item for item in items if item["evidence_id"] not in removed_ids]
        params["removed_ids"] = sorted(removed_ids)
        validation = "requires_human_validation"
    elif transformation == "contradiction_insertion":
        additions = deepcopy(distractors or [])
        if not additions or any(item.get("relation") != "CONTRADICT" for item in additions):
            raise ValueError("Contradiction insertion requires CONTRADICT records.")
        existing = {item["evidence_id"] for item in items}
        if any(item.get("evidence_id") in existing for item in additions):
            raise ValueError("Inserted evidence IDs must be unique.")
        items.extend(additions)
        params["inserted_ids"] = [item["evidence_id"] for item in additions]
        validation = "requires_human_validation"
    else:
        raise ValueError(f"Unsupported transformation: {transformation}")

    status, condition = _state(items)
    result["evidence_items"] = items
    result["original_example_id"] = example.get("original_example_id", example["example_id"])
    result["transformation_id"] = f"{example['example_id']}::{transformation}::{seed}"
    result["transformation_type"] = transformation
    result["original_evidence_ids"] = [item["evidence_id"] for item in original]
    result["transformed_evidence_ids"] = [item["evidence_id"] for item in items]
    result["transformation_parameters"] = params
    result["original_claim_status"] = example.get("claim_status", _state(original)[0])
    result["claim_status"], result["evidence_condition"] = status, condition
    result["label_validation_status"] = validation
    result["random_seed"] = seed
    return result
