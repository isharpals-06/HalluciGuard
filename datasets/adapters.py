"""Local-file adapters that retain source annotations and dataset splits.

These adapters never download data or write transformed copies. Callers must place
permitted benchmark files locally and review each dataset's current licence.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


TRIVIAPLUS_LABEL_MAP = {
    "supports": "SUPPORTED",
    "contradicts": "CONTRADICTED",
    "not-mentioned": "INSUFFICIENT_EVIDENCE",
    # Supplementary is intentionally kept outside the three-way claim taxonomy.
    "supplementary": "SUPPLEMENTARY",
}


def adapt_triviaplus_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize TRIVIA+ sentence annotations while preserving original labels.

    Rows must be supplied by a caller (for example, via an installed Parquet
    engine). No sentence/label alignment is guessed or repaired here.
    """
    output = []
    for row_number, row in enumerate(rows):
        sentences = row.get("answer_sentence_list")
        labels = row.get("sentence_level_majority_vote")
        if not isinstance(sentences, (list, tuple)) or not isinstance(labels, (list, tuple)):
            raise ValueError(f"TRIVIA+ row {row_number} lacks sentence list or majority-vote labels.")
        if len(sentences) != len(labels):
            raise ValueError(f"TRIVIA+ row {row_number} has misaligned sentence and label arrays.")
        converted = []
        for index, (sentence, label) in enumerate(zip(sentences, labels)):
            source_label = str(label).strip().lower()
            if source_label not in TRIVIAPLUS_LABEL_MAP:
                raise ValueError(f"Unknown TRIVIA+ sentence label: {label!r}")
            converted.append({
                "claim_id": f"{row.get('id', row_number)}:sentence:{index}",
                "claim_text": str(sentence),
                "source_label": label,
                "claim_status": TRIVIAPLUS_LABEL_MAP[source_label],
                "split": row.get("split"),
            })
        output.append({
            "dataset": "TRIVIA+",
            "example_id": str(row.get("id", row_number)),
            "split": row.get("split"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "context": row.get("article"),
            "claims": converted,
        })
    return output


def load_triviaplus(parquet_path: str | Path) -> List[Dict[str, Any]]:
    """Load a user-provided local Parquet release with optional pandas/pyarrow."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "TRIVIA+ Parquet loading requires the optional pandas and pyarrow packages. "
            "Install them only in an environment permitted to access the dataset."
        ) from exc
    frame = pd.read_parquet(parquet_path)
    return adapt_triviaplus_rows(frame.to_dict(orient="records"))


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def load_ragtruth(responses_path: str | Path, source_info_path: str | Path) -> List[Dict[str, Any]]:
    """Join RAGTruth response spans to source records without span-to-claim mapping."""
    source_records = _read_jsonl(source_info_path)
    sources = {str(record["source_id"]): record for record in source_records}
    if len(sources) != len(source_records):
        raise ValueError("RAGTruth source_info contains duplicate source_id values.")
    output = []
    for response in _read_jsonl(responses_path):
        source_id = str(response.get("source_id", ""))
        if source_id not in sources:
            raise ValueError(f"RAGTruth response references unknown source_id {source_id!r}.")
        output.append({
            "dataset": "RAGTruth",
            "example_id": str(response["id"]),
            "source_id": source_id,
            "split": response.get("split"),
            "task_type": sources[source_id].get("task_type"),
            "query_or_prompt": sources[source_id].get("source_info"),
            "response": response.get("response", ""),
            "hallucination_spans": response.get("labels", []),
            "quality": response.get("quality"),
            "source_labels_preserved": True,
            "claim_alignment": "not_performed",
        })
    return output
