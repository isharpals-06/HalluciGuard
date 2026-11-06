"""
Freshness and Temporal Validity Verification Service.
Evaluates the temporal validity of retrieved OKF evidence relative to claims.
Assigns discrete state: CURRENT, HISTORICAL, STALE, UNKNOWN.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import re
from okf.schemas.okf_spec import OKFArtifact


class FreshnessAnalyzer:
    def __init__(self, reference_time_iso: Optional[str] = None):
        if reference_time_iso:
            self.reference_time = datetime.fromisoformat(reference_time_iso.replace("Z", "+00:00"))
        else:
            self.reference_time = datetime.now(timezone.utc)

    def evaluate(self, claim_text: str, artifact: OKFArtifact) -> Dict[str, Any]:
        """
        Evaluates temporal consistency between claim and OKF artifact.
        Returns:
            {
                "freshness_state": "CURRENT" | "HISTORICAL" | "STALE" | "UNKNOWN",
                "freshness_confidence": float [0, 1],
                "is_stale": bool
            }
        """
        temp = artifact.temporal
        lifecycle = artifact.lifecycle

        # Check claim for "current" tense or assertion
        claim_implies_current = bool(
            re.search(r"\b(current|currently|now|today|presently|is the current|serves as)\b", claim_text, re.IGNORECASE)
        )

        valid_until_dt = None
        if temp.valid_until:
            try:
                valid_until_dt = datetime.fromisoformat(temp.valid_until.replace("Z", "+00:00"))
            except Exception:
                pass

        valid_from_dt = None
        if temp.valid_from:
            try:
                valid_from_dt = datetime.fromisoformat(temp.valid_from.replace("Z", "+00:00"))
            except Exception:
                pass

        # Case 1: Artifact has an expired valid_until, but claim claims it is currently true
        if valid_until_dt and valid_until_dt < self.reference_time:
            if claim_implies_current:
                # Stale assertion!
                return {
                    "freshness_state": "STALE",
                    "freshness_confidence": 0.95,
                    "is_stale": True,
                }
            else:
                # Historical fact stated in past tense
                return {
                    "freshness_state": "HISTORICAL",
                    "freshness_confidence": 0.92,
                    "is_stale": False,
                }

        # Case 2: Explicit permanent or currently valid scope
        if temp.temporal_scope in ("permanent", "current") or (valid_until_dt is None and valid_from_dt):
            return {
                "freshness_state": "CURRENT",
                "freshness_confidence": 0.98,
                "is_stale": False,
            }

        # Case 3: Inherit from lifecycle state
        state = lifecycle.freshness_state
        conf = 0.90 if state != "UNKNOWN" else 0.50
        is_stale = (state == "STALE")

        return {
            "freshness_state": state,
            "freshness_confidence": conf,
            "is_stale": is_stale,
        }
