"""
Natural Language Inference (NLI) Verification Engine.
Computes calibrated probabilities for Entailment, Neutral, and Contradiction
given premise (evidence) and hypothesis (claim).
Features robust semantic grounding, subject-attribute relation alignment, and optional transformer pipeline.
"""

from typing import Dict, Optional, Tuple
import re
import numpy as np


class NLIEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 1.0,
        use_transformer: bool = False,
        mode: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = max(0.1, temperature)
        self.mode = mode or ("transformer" if use_transformer else "rule_based")
        if self.mode not in {"transformer", "rule_based"}:
            raise ValueError("NLI mode must be 'transformer' or 'rule_based'.")
        self.use_transformer = self.mode == "transformer"
        self.pipeline = None

        if self.use_transformer and not self.model_name:
            raise ValueError("A model_name is required for transformer NLI mode.")
        if self.use_transformer:
            self._load_transformer()

    def _load_transformer(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                top_k=None,
            )
        except Exception as e:
            raise RuntimeError(
                f"Transformer NLI mode could not load {self.model_name!r}; "
                "select mode='rule_based' explicitly to use heuristic scoring."
            ) from e

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculates P(Entailment), P(Neutral), P(Contradiction) for (premise, hypothesis).
        """
        if not premise or not hypothesis:
            return {"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05}

        if self.mode == "transformer":
            try:
                outputs = self.pipeline({"text": premise, "text_pair": hypothesis})[0]
                probs = {}
                for out in outputs:
                    label = out["label"].lower()
                    if "entail" in label:
                        probs["entailment"] = out["score"]
                    elif "contra" in label:
                        probs["contradiction"] = out["score"]
                    elif "neut" in label:
                        probs["neutral"] = out["score"]
                if set(probs) != {"entailment", "neutral", "contradiction"}:
                    raise RuntimeError(f"Unexpected NLI labels returned by {self.model_name}: {list(probs)}")
                return self._apply_temperature(probs)
            except Exception as e:
                raise RuntimeError(f"Transformer NLI inference failed for {self.model_name!r}.") from e

        return self._rule_based_nli(premise, hypothesis)

    def _rule_based_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Calibrated semantic, temporal, numerical, and relational NLI engine.
        Enforces topical grounding and structured subject-attribute relation alignment.
        """
        p_clean = premise.lower()
        h_clean = hypothesis.lower()

        p_words = set(re.findall(r"\w+", p_clean))
        h_words = set(re.findall(r"\w+", h_clean))

        stopwords = {
            "the", "a", "an", "is", "was", "were", "are", "in", "on", "at", "by", "of",
            "to", "and", "or", "for", "with", "as", "it", "from", "that", "this", "be",
            "has", "had", "have", "its", "their", "he", "she", "dr"
        }
        informative_h = h_words - stopwords
        informative_p = p_words - stopwords

        if not informative_h:
            return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

        common_informative = informative_h.intersection(informative_p)
        overlap = len(common_informative) / len(informative_h) if informative_h else 0.0

        # Topical grounding guard: If premise and hypothesis share no core entities/terms,
        # they are about completely different subjects -> Neutral (Insufficient Evidence)
        if len(common_informative) < 1 and overlap < 0.15:
            return {"entailment": 0.03, "neutral": 0.94, "contradiction": 0.03}

        # 1. Double negation resolution (e.g. "not untrue", "not false", "not impossible")
        double_neg_pattern = r"\bnot\s+(?:untrue|false|incorrect|unlikely|impossible)\b"
        if re.search(double_neg_pattern, h_clean):
            h_resolved = re.sub(double_neg_pattern, "", h_clean)
            resolved_h_words = set(re.findall(r"\w+", h_resolved)) - stopwords
            resolved_overlap = len(resolved_h_words.intersection(informative_p)) / len(resolved_h_words) if resolved_h_words else 0.0
            if resolved_overlap >= 0.40:
                return {"entailment": 0.92, "neutral": 0.05, "contradiction": 0.03}

        # 2. Subject-Attribute-Value Relation Clash Check:
        # Matches patterns:
        # P: "<SubjectA> is (the)? <Attribute> of <Domain>" or "<Attribute> of <Domain> is <SubjectA>"
        # H: "<SubjectB> is (the)? <Attribute> of <Domain>" or "<Attribute> of <Domain> is <SubjectB>"
        pat_subj_attr = r"([a-z]+)\s+is\s+(?:the\s+)?(?:official\s+)?([a-z\s]+?)\s+of\s+([a-z]+)"
        pat_attr_subj = r"(?:the\s+)?([a-z\s]+?)\s+of\s+([a-z]+)\s+is\s+([a-z]+)"

        rel_p = None
        m_p1 = re.search(pat_subj_attr, p_clean)
        m_p2 = re.search(pat_attr_subj, p_clean)
        if m_p1:
            rel_p = {"subj": m_p1.group(1).strip(), "attr": m_p1.group(2).strip(), "domain": m_p1.group(3).strip()}
        elif m_p2:
            rel_p = {"subj": m_p2.group(3).strip(), "attr": m_p2.group(1).strip(), "domain": m_p2.group(2).strip()}

        rel_h = None
        m_h1 = re.search(pat_subj_attr, h_clean)
        m_h2 = re.search(pat_attr_subj, h_clean)
        if m_h1:
            rel_h = {"subj": m_h1.group(1).strip(), "attr": m_h1.group(2).strip(), "domain": m_h1.group(3).strip()}
        elif m_h2:
            rel_h = {"subj": m_h2.group(3).strip(), "attr": m_h2.group(1).strip(), "domain": m_h2.group(2).strip()}

        if rel_p and rel_h:
            # Check if both refer to the same domain and attribute
            if rel_p["domain"] == rel_h["domain"] and (rel_p["attr"] in rel_h["attr"] or rel_h["attr"] in rel_p["attr"]):
                if rel_p["subj"] != rel_h["subj"]:
                    return {"entailment": 0.01, "neutral": 0.04, "contradiction": 0.95}
                else:
                    return {"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01}

        # 3. Temporal clash check (Only if topically grounded with overlap >= 0.20 or entity match)
        p_years = set(re.findall(r"\b(1[789]\d{2}|20\d{2})\b", p_clean))
        h_years = set(re.findall(r"\b(1[789]\d{2}|20\d{2})\b", h_clean))

        if p_years and h_years and (overlap >= 0.20 or len(common_informative) >= 2):
            if not p_years.intersection(h_years):
                return {"entailment": 0.02, "neutral": 0.06, "contradiction": 0.92}

        # 4. Number contradiction check (Only if topically grounded)
        p_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", p_clean)) - p_years
        h_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", h_clean)) - h_years
        if p_nums and h_nums and (overlap >= 0.25 or len(common_informative) >= 2):
            if not p_nums.intersection(h_nums):
                return {"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90}

        # 5. Attribution mismatch check:
        p_by = re.findall(r"by\s+([A-Za-z\s]+?)(?:in|\.|,|$)", premise)
        h_by = re.findall(r"by\s+([A-Za-z\s]+?)(?:in|\.|,|$)", hypothesis)
        if p_by and h_by and overlap >= 0.20:
            p_names = set(p_by[0].lower().split())
            h_names = set(h_by[0].lower().split()) - {"the", "a"}
            if not h_names.intersection(p_names):
                return {"entailment": 0.02, "neutral": 0.06, "contradiction": 0.92}

        # 6. Standard Single Negation clash:
        negations = ["not", "never", "failed", "refused", "no longer", "unable"]
        p_neg = any(f" {n} " in f" {p_clean} " for n in negations)
        h_neg = any(f" {n} " in f" {h_clean} " for n in negations)
        if p_neg != h_neg and len(common_informative) >= 3:
            return {"entailment": 0.03, "neutral": 0.07, "contradiction": 0.90}

        # 7. Semantic & Lexical Entailment based on overlap:
        if overlap >= 0.65:
            p_ent = min(0.96, 0.45 + overlap * 0.55)
            p_con = max(0.01, 0.03 * (1.0 - overlap))
            p_neu = max(0.01, 1.0 - p_ent - p_con)
            return {"entailment": float(p_ent), "neutral": float(p_neu), "contradiction": float(p_con)}
        elif overlap >= 0.35:
            return {"entailment": 0.35, "neutral": 0.55, "contradiction": 0.10}
        else:
            return {"entailment": 0.05, "neutral": 0.88, "contradiction": 0.07}

    def _apply_temperature(self, probs: Dict[str, float]) -> Dict[str, float]:
        labels = ["entailment", "neutral", "contradiction"]
        logits = np.array([np.log(max(1e-7, probs.get(l, 0.33))) for l in labels])
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)
        return {l: float(p) for l, p in zip(labels, calibrated_probs)}
