"""Lightweight retrieval for the Retrieval Selection technique: split a
document into its paragraphs, embed each one, and keep only the top-k
most similar to the question. Deliberately the leanest possible form of
retrieval, a single dense search, no fusion or reranking. This
experiment isn't about which retrieval architecture wins (see the
separate rag-architectures experiment for that); it's about whether
choosing *what to keep in context* by relevance beats keeping everything
or keeping only what's recent.

`from __future__ import annotations` defers type hints to strings, so
importing this module doesn't require sentence-transformers to be
installed, only actually calling select_relevant_paragraphs() does. That
keeps the package out of the dashboard's requirements.txt.
"""

from __future__ import annotations

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def select_relevant_paragraphs(document: str, question: str, k: int) -> list[str]:
    paragraphs = [p for p in document.split("\n\n") if p.strip()]
    if len(paragraphs) <= k:
        return paragraphs

    embedder = get_embedder()
    paragraph_embeddings = embedder.encode(paragraphs, normalize_embeddings=True, show_progress_bar=False)
    query_embedding = embedder.encode([question], normalize_embeddings=True)[0]
    scores = np.asarray(paragraph_embeddings) @ query_embedding
    top_indices = np.argsort(-scores)[:k]

    # Keep the retrieved paragraphs in their original document order, not
    # ranked by score: it reads as one coherent excerpt to the generator
    # rather than a shuffled bag of the k best-scoring pieces.
    return [paragraphs[i] for i in sorted(top_indices)]
