"""Embedding-based few-shot example retrieval and a semantic response cache.

Both run on a local sentence-transformer rather than an API embedding call,
so retrieval cost doesn't confound the operational metrics we're actually
trying to measure (those belong to the classification call itself).
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

MIN_RELEVANT_SIMILARITY = 0.3  # below this, a training example isn't a fair few-shot demo
CACHE_DUPLICATE_SIMILARITY = 0.97  # above this, two queries are near-duplicates

_embedder = SentenceTransformer(EMBEDDING_MODEL_ID)


def embed(texts: list[str]) -> np.ndarray:
    return _embedder.encode(texts, normalize_embeddings=True)


@dataclass
class RetrievalResult:
    examples: list[dict]
    is_empty: bool  # true when nothing in the pool was relevant enough to show


class ExampleRetriever:
    """k-nearest-neighbor lookup over the training pool, by embedding similarity."""

    def __init__(self, train_examples: list[dict]):
        self.train_examples = train_examples
        self.train_embeddings = embed([example["text"] for example in train_examples])

    def retrieve(self, query_text: str, k: int = 5) -> RetrievalResult:
        query_embedding = embed([query_text])[0]
        similarities = self.train_embeddings @ query_embedding

        if similarities.max() < MIN_RELEVANT_SIMILARITY:
            return RetrievalResult(examples=[], is_empty=True)

        top_k_indices = np.argsort(similarities)[::-1][:k]
        return RetrievalResult(
            examples=[self.train_examples[i] for i in top_k_indices],
            is_empty=False,
        )


class SemanticCache:
    """Reuses a prior prediction when a new query is a near-duplicate of one
    already answered, instead of calling the model again."""

    def __init__(self):
        self._embeddings: list[np.ndarray] = []
        self._predictions: list[dict] = []

    def lookup(self, query_text: str) -> dict | None:
        if not self._embeddings:
            return None
        query_embedding = embed([query_text])[0]
        similarities = np.stack(self._embeddings) @ query_embedding
        best_index = int(similarities.argmax())
        if similarities[best_index] >= CACHE_DUPLICATE_SIMILARITY:
            return self._predictions[best_index]
        return None

    def store(self, query_text: str, prediction: dict) -> None:
        self._embeddings.append(embed([query_text])[0])
        self._predictions.append(prediction)
