"""
BM25 Sparse Retrieval Service.
Performs lexical matching of claims against OKF knowledge artifacts.
Includes IDF smoothing to handle small-corpus edge cases.
"""

import math
import re
from typing import List, Tuple
from okf.schemas.okf_spec import OKFArtifact


class BM25Retriever:
    def __init__(self, artifacts: List[OKFArtifact]):
        self.artifacts = artifacts
        self.corpus = [self._tokenize(art.to_summary_text()) for art in artifacts]
        self.avg_doc_len = (
            sum(len(d) for d in self.corpus) / len(self.corpus) if self.corpus else 1.0
        )
        self.doc_freqs = self._calc_doc_freqs()

    STOPWORDS = {
        "the", "a", "an", "is", "was", "were", "in", "on", "at", "by", "of", "to", "and", "or", "for", "with", "as", "it", "from", "that", "this", "be", "has", "had"
    }

    def _tokenize(self, text: str) -> List[str]:
        tokens = [w.lower() for w in re.findall(r"\w+", text)]
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]

    def _calc_doc_freqs(self):
        df = {}
        for doc in self.corpus:
            unique_terms = set(doc)
            for t in unique_terms:
                df[t] = df.get(t, 0) + 1
        return df

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[OKFArtifact, float]]:
        """
        Retrieves top_k artifacts using smoothed Okapi BM25.
        """
        if not self.corpus or not self.artifacts:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        N = len(self.corpus)
        k1 = 1.5
        b = 0.75

        # Compute theoretical ideal query score for normalization (all query terms present)
        ideal_query_score = 0.0
        unique_q_tokens = set(q_tokens)
        for q in unique_q_tokens:
            df = self.doc_freqs.get(q, 0)
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5)) + 0.5
            ideal_query_score += idf * (k1 + 1.0)
        if ideal_query_score <= 0.0:
            ideal_query_score = 1.0

        scores = []
        for i, doc in enumerate(self.corpus):
            doc_len = len(doc)
            score = 0.0
            doc_term_counts = {}
            for t in doc:
                doc_term_counts[t] = doc_term_counts.get(t, 0) + 1

            for q in q_tokens:
                if q in doc_term_counts:
                    freq = doc_term_counts[q]
                    df = self.doc_freqs.get(q, 0)
                    # Smoothed IDF ensuring strictly positive values
                    idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5)) + 0.5
                    tf = (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len)))
                    score += idf * tf

            scores.append(score)

        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            raw_s = scores[idx]
            if raw_s > 0:
                norm_score = min(1.0, max(0.0, raw_s / ideal_query_score))
                if norm_score > 0.05:
                    results.append((self.artifacts[idx], float(norm_score)))

        return results
