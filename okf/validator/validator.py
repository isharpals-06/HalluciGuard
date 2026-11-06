"""
OKF Schema and Semantic Validator.
Validates candidate knowledge artifacts against the official OKF v1.0 standard.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
from pydantic import ValidationError
from okf.schemas.okf_spec import OKFArtifact


class OKFValidator:
    def __init__(self, schema_path: str = "configs/okf_schema.json"):
        self.schema_path = Path(schema_path)
        self.raw_schema = None
        if self.schema_path.exists():
            with open(self.schema_path, "r", encoding="utf-8") as f:
                self.raw_schema = json.load(f)

    def validate(self, artifact_dict: Dict[str, Any]) -> Tuple[bool, List[str], OKFArtifact]:
        """
        Validates an OKF artifact dictionary.
        Returns:
            (is_valid, error_list, parsed_artifact_or_none)
        """
        errors = []
        parsed = None

        if not isinstance(artifact_dict, dict):
            return False, ["Artifact must be a JSON dictionary object"], None

        # 1. Pydantic validation
        try:
            parsed = OKFArtifact(**artifact_dict)
        except ValidationError as e:
            for err in e.errors():
                loc = " -> ".join(str(p) for p in err["loc"])
                errors.append(f"Validation error at '{loc}': {err['msg']}")
            return False, errors, None
        except Exception as e:
            return False, [f"Unexpected parse error: {str(e)}"], None

        # 2. Semantic Integrity Checks
        # Temporal consistency: if valid_from and valid_until are set, valid_until must be >= valid_from
        t = parsed.temporal
        if t.valid_from and t.valid_until:
            if t.valid_until < t.valid_from:
                errors.append(
                    f"Temporal anomaly: valid_until ({t.valid_until}) precedes valid_from ({t.valid_from})"
                )

        # Reliability check
        if parsed.source.reliability_estimate < 0.0 or parsed.source.reliability_estimate > 1.0:
            errors.append("source.reliability_estimate must be within [0.0, 1.0]")

        # Verification confidence check
        if parsed.verification.confidence < 0.0 or parsed.verification.confidence > 1.0:
            errors.append("verification.confidence must be within [0.0, 1.0]")

        is_valid = len(errors) == 0
        return is_valid, errors, parsed if is_valid else None
