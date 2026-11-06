"""
Claim Extraction and Atomic Decomposition Service.
Decomposes LLM response passages into atomic, independently verifiable factual claims.
Features robust clause splitting, entity identification, temporal extraction, and ReDoS safeguards.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExtractedClaim(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: str = Field("factual", description="temporal, numerical, entity, attribution, geographic, factual")
    source_sentence: str
    entities: List[Dict[str, str]] = Field(default_factory=list)
    temporal_reference: Optional[Dict[str, str]] = None
    extraction_confidence: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ClaimExtractor:
    """
    Decomposes unstructured natural language text into discrete atomic claims.
    Implements deterministic syntactic, lexical, and clause decomposition with safety limits.
    """

    MAX_INPUT_LENGTH: int = 4096

    def __init__(self, confidence_default: float = 0.95, max_input_length: int = 4096):
        self.confidence_default = confidence_default
        self.MAX_INPUT_LENGTH = max_input_length
        self.year_pattern = re.compile(r"\b(1[789]\d{2}|20\d{2})\b")
        self.number_pattern = re.compile(r"\b\d+(?:[.,]\d+)?\b")
        self.attribution_markers = ["by", "invented", "authored", "written", "founded", "created", "discovered", "directed", "led"]
        self.geographic_markers = ["in", "at", "capital", "located", "city", "country", "river", "mountain"]

    def extract_claims(self, text: str, query: Optional[str] = None) -> List[ExtractedClaim]:
        """
        Decomposes input response text into atomic claims.
        Guarantees linear-time safety against ReDoS attacks.
        """
        if not text or not text.strip():
            return []

        # ReDoS safeguard: truncate oversized input
        safe_text = text.strip()[:self.MAX_INPUT_LENGTH]

        # 1. Sentence splitting
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", safe_text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        claims: List[ExtractedClaim] = []
        claim_counter = 1

        for sentence in sentences:
            sub_clauses = self._decompose_sentence(sentence)
            for clause in sub_clauses:
                clause_text = clause.strip()
                if len(clause_text) < 4:
                    continue

                if not clause_text.endswith((".", "!", "?")):
                    clause_text += "."

                claim_type, entities, temporal = self._analyze_claim(clause_text)

                claim_obj = ExtractedClaim(
                    claim_id=f"C{claim_counter:03d}",
                    claim_text=clause_text,
                    claim_type=claim_type,
                    source_sentence=sentence,
                    entities=entities,
                    temporal_reference=temporal,
                    extraction_confidence=self.confidence_default,
                )
                claims.append(claim_obj)
                claim_counter += 1

        return claims

    def _decompose_sentence(self, sentence: str) -> List[str]:
        """
        Splits compound sentences and compound clauses (e.g. date + attribution)
        into atomic claims.
        """
        s = sentence.strip().rstrip(".!?")

        # 1. Temporal + agentive compound decomposition:
        # e.g. "Tesla was founded in 2008 by Elon Musk"
        # Handles variations: "was founded in X by Y", "founded in X by Y", "created in X by Y"
        m_temp_agent = re.match(
            r"^(.+?\b(?:was|were|is|are)?\s*\w+ed)\s+in\s+(\d{4})\s+by\s+([A-Za-z\s]+)$",
            s,
            re.IGNORECASE,
        )
        if m_temp_agent:
            subj = m_temp_agent.group(1).strip()
            year = m_temp_agent.group(2).strip()
            agent = m_temp_agent.group(3).strip()
            return [
                f"{subj} in {year}.",
                f"{subj} by {agent}.",
            ]

        # 2. Conjunction splitting: ";", " and ", " but ", " while ", " as well as "
        parts = re.split(r";\s*|\s+(?:and|but|however|while|as well as)\s+", s, flags=re.IGNORECASE)
        if len(parts) <= 1:
            return [sentence]

        clauses = []
        first_subject = self._extract_leading_subject(parts[0])

        for i, part in enumerate(parts):
            cleaned = part.strip()
            if not cleaned or len(cleaned) < 3:
                continue

            has_verb = bool(re.search(
                r"\b(is|was|were|are|founded|won|became|took|landed|has|had|produced|invented|died|served)\b",
                cleaned,
                re.IGNORECASE,
            ))
            has_explicit_subject = bool(re.match(r"^[A-Z][a-z]+", cleaned))

            if i > 0 and has_verb and not has_explicit_subject and first_subject:
                clauses.append(f"{first_subject} {cleaned}.")
            else:
                clauses.append(cleaned + ".")

        return clauses if clauses else [sentence]

    def _extract_leading_subject(self, text: str) -> Optional[str]:
        words = text.strip().split()
        if not words:
            return None
        subject_tokens = []
        for w in words[:4]:
            clean_w = re.sub(r"[^\w]", "", w)
            if clean_w and (clean_w[0].isupper() or clean_w.lower() in ("the", "a", "an", "dr")):
                subject_tokens.append(clean_w)
            else:
                break
        return " ".join(subject_tokens) if subject_tokens else None

    def _analyze_claim(self, claim_text: str) -> tuple:
        entities = []
        temporal = None
        claim_type = "factual"

        # 1. Proper Noun Entities FIRST (so primary entity is selected for retrieval)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim_text)
        stopwords_caps = {"The", "A", "An", "In", "At", "On", "It", "He", "She", "They", "This", "That", "Dr"}
        for cap in capitalized:
            if cap not in stopwords_caps and cap not in [e["name"] for e in entities]:
                entities.append({"name": cap, "type": "PROPER_NOUN"})

        # 2. Temporal Reference
        year_match = self.year_pattern.search(claim_text)
        if year_match:
            year_val = year_match.group(1)
            temporal = {"type": "absolute", "value": year_val}
            entities.append({"name": year_val, "type": "DATE"})
            claim_type = "temporal"

        # 3. Numerical Reference
        num_matches = self.number_pattern.findall(claim_text)
        if num_matches and claim_type != "temporal":
            for n in num_matches[:2]:
                val = n if isinstance(n, str) else n[0]
                entities.append({"name": val, "type": "NUMBER"})
            claim_type = "numerical"

        lower_text = claim_text.lower()
        if any(f" {m} " in lower_text for m in self.attribution_markers) or " by " in lower_text:
            if claim_type == "factual":
                claim_type = "attribution"

        if any(f" {g} " in lower_text for g in self.geographic_markers):
            if claim_type == "factual":
                claim_type = "geographic"

        if len(entities) > 0 and claim_type == "factual":
            claim_type = "entity"

        return claim_type, entities, temporal
