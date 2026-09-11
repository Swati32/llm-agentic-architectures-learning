"""The retrieval layer every architecture shares: one global index built
once over the whole chunked corpus, reused across every query. That's a
deliberate difference from this repo's agentic-architectures experiment,
where each question got its own tiny 10-paragraph corpus — here the corpus
is the same ~600-article MultiHop-RAG collection for every query, which is
what actually makes chunking (how a corpus is cut before indexing, not how
one already-short passage is used) a variable worth studying at all.

`from __future__ import annotations` defers every type hint in this file
to a string, so importing this module doesn't require sentence-transformers
or rank_bm25 to be installed, only actually calling build_index() does.
That's what keeps those packages out of the dashboard's requirements.txt.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import numpy as np

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray  # (n_chunks, dim), L2-normalized
    bm25: "BM25Okapi"


_WORD_RE = re.compile(r"\w+")


def _tokenize_for_bm25(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def build_index(articles: list[dict], chunk_fn) -> Index:
    """chunk_fn(text, embedder) -> list[str], one of chunking/strategies.py's
    STRATEGIES. Passing the embedder through lets chunk_semantic reuse the
    same loaded model instead of loading a second copy."""
    from rank_bm25 import BM25Okapi

    embedder = get_embedder()
    chunks: list[Chunk] = []
    for article in articles:
        pieces = chunk_fn(article["body"], embedder)
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(chunk_id=f"{article['title']}::{i}", title=article["title"], text=piece))

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    bm25 = BM25Okapi([_tokenize_for_bm25(text) for text in texts])
    return Index(chunks=chunks, embeddings=np.asarray(embeddings), bm25=bm25)


@dataclass
class RetrievedChunk:
    chunk_id: str
    title: str
    text: str
    score: float


def dense_search(index: Index, query: str, k: int) -> list[RetrievedChunk]:
    embedder = get_embedder()
    query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
    scores = index.embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:k]
    return [
        RetrievedChunk(index.chunks[i].chunk_id, index.chunks[i].title, index.chunks[i].text, float(scores[i]))
        for i in top_indices
    ]


def dense_search_by_vector(index: Index, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
    """Same as dense_search, but takes an already-computed query embedding.
    HyDE uses this: it embeds a hypothetical answer passage instead of the
    raw query, so the embedding step happens outside this function."""
    scores = index.embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:k]
    return [
        RetrievedChunk(index.chunks[i].chunk_id, index.chunks[i].title, index.chunks[i].text, float(scores[i]))
        for i in top_indices
    ]


def bm25_search(index: Index, query: str, k: int) -> list[RetrievedChunk]:
    scores = index.bm25.get_scores(_tokenize_for_bm25(query))
    top_indices = np.argsort(-scores)[:k]
    return [
        RetrievedChunk(index.chunks[i].chunk_id, index.chunks[i].title, index.chunks[i].text, float(scores[i]))
        for i in top_indices
    ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int, rrf_constant: int = 60
) -> list[RetrievedChunk]:
    """Merges two or more independently-ranked lists into one ranking,
    using each chunk's rank (position), not its raw score, so a dense
    cosine score and a BM25 score never need to be on the same scale.
    Standard formula from Cormack et al., 2009:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_constant + rank + 1)
            chunk_by_id[chunk.chunk_id] = chunk
    ranked_ids = sorted(fused_scores, key=lambda cid: -fused_scores[cid])[:k]
    return [
        RetrievedChunk(chunk_by_id[cid].chunk_id, chunk_by_id[cid].title, chunk_by_id[cid].text, fused_scores[cid])
        for cid in ranked_ids
    ]


def rerank(query: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    """Cross-encoder reranking: unlike dense_search's cosine similarity,
    which embeds the query and each chunk independently, a cross-encoder
    reads the query and a chunk together in one forward pass, so it can
    catch relevance dense search misses at the cost of being too slow to
    run over the whole corpus (hence: rerank a shortlist, don't replace
    the first-pass retrieval)."""
    reranker = get_reranker()
    pairs = [(query, chunk.text) for chunk in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(-scores)[:k]
    return [
        RetrievedChunk(candidates[i].chunk_id, candidates[i].title, candidates[i].text, float(scores[i]))
        for i in order
    ]


def _normalize(text: str) -> str:
    text = text.lower()
    return "".join(ch for ch in text if ch not in string.punctuation)


def is_gold_chunk(chunk_text: str, evidence_facts: list[str]) -> bool:
    """A chunk counts as gold if it contains one of the query's evidence
    facts as a (near-)substring. Facts are verbatim sentences lifted from
    the source article body, so this holds regardless of where a given
    chunking strategy happened to draw its boundaries — which is what
    makes Recall@k comparable across chunking strategies in the first
    place, not just across architectures."""
    if not evidence_facts:
        return False
    normalized_chunk = _normalize(chunk_text)
    return any(_normalize(fact) in normalized_chunk for fact in evidence_facts)
