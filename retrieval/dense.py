"""
Dense Semantic Vector Retrieval Service.
Encodes OKF knowledge statements and claims into dense representation spaces.
"""

from typing import List, Tuple, Optional
import numpy as np
from okf.schemas.okf_spec import OKFArtifact


class DenseRetriever:
    def __init__(
        self,
        artifacts: List[OKFArtifact],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_fallback: bool = True,
        mode: Optional[str] = None,
    ):
        self.artifacts = artifacts
        self.model_name = model_name
        self.mode = mode or ("tfidf" if use_fallback else "sentence_transformer")
        if self.mode not in {"tfidf", "sentence_transformer"}:
            raise ValueError("Dense retrieval mode must be 'tfidf' or 'sentence_transformer'.")
        self.use_fallback = self.mode == "tfidf"
        self.model = None
        self.corpus_embeddings = None
        self.fallback_vectorizer = None

        self._initialize_index()

    def _initialize_index(self):
        if not self.artifacts:
            return

        corpus_texts = [art.content.statement for art in self.artifacts]

        if self.mode == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
                self.corpus_embeddings = embeddings
                return
            except Exception as e:
                raise RuntimeError(
                    f"SentenceTransformer mode could not load {self.model_name!r}; "
                    "select mode='tfidf' explicitly to use the lexical fallback."
                ) from e

        # Fallback dense representation using TF-IDF with English stop words + normalization
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.fallback_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=512)
        tfidf_mat = self.fallback_vectorizer.fit_transform(corpus_texts).toarray()
        # L2 normalize
        norms = np.linalg.norm(tfidf_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.corpus_embeddings = tfidf_mat / norms

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[OKFArtifact, float]]:
        if self.corpus_embeddings is None or not self.artifacts:
            return []

        if self.mode == "sentence_transformer" and self.model:
            query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            q_vec = self.fallback_vectorizer.transform([query]).toarray()[0]
            norm = np.linalg.norm(q_vec)
            query_emb = q_vec / norm if norm > 0 else q_vec

        # Cosine similarity (already L2 normalized)
        sims = np.dot(self.corpus_embeddings, query_emb)

        ranked_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(sims[idx])
            if score > 0.05:
                results.append((self.artifacts[idx], max(0.0, min(1.0, score))))

        return results
