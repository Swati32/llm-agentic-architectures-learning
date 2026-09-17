"""The one tool every architecture shares: semantic search over a fixed,
per-question corpus of 10 candidate paragraphs (2 gold, 8 distractors).

Embeddings are computed once per question with a small local
sentence-transformer and reused across all five architectures, so retrieval
quality is identical everywhere and any accuracy difference between
architectures comes from how they plan and use the tool, not from
differences in the tool itself.

Returning only the single best-matching paragraph (k=1) is deliberate: with
2 gold paragraphs needed per question, a single search can never fully
answer a question, forcing every architecture to actually decide whether
and how to search again.

`from __future__ import annotations` below defers every type hint in this
file to a string, so importing this module doesn't require
sentence-transformers to be installed at all, only actually calling
build_corpus() or search() does. That's what keeps sentence-transformers
out of the dashboard's requirements.txt: app.py imports this module
transitively (through techniques/) but never calls either function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedder: "SentenceTransformer | None" = None


def _get_embedder() -> "SentenceTransformer":
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


@dataclass
class Corpus:
    paragraphs: list[dict]  # [{"title": ..., "text": ...}, ...]
    embeddings: np.ndarray
    gold_titles: set[str]


def build_corpus(example: dict) -> Corpus:
    embedder = _get_embedder()
    texts = [f"{p['title']}. {p['text']}" for p in example["paragraphs"]]
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    return Corpus(paragraphs=example["paragraphs"], embeddings=embeddings, gold_titles=example["gold_titles"])


def search(corpus: Corpus, query: str) -> dict:
    embedder = _get_embedder()
    query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
    scores = corpus.embeddings @ query_embedding
    best_index = int(np.argmax(scores))
    paragraph = corpus.paragraphs[best_index]
    return {
        "title": paragraph["title"],
        "text": paragraph["text"],
        "score": float(scores[best_index]),
        "is_gold": paragraph["title"] in corpus.gold_titles,
    }
