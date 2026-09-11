"""Five ways to split a long article into retrievable chunks. This is the
one variable the chunking sub-experiment isolates: every strategy runs
against the same corpus, the same embedder, and the same architecture
(naive RAG), so any difference in retrieval or answer quality comes from
how the text was cut, not from anything else.

Token counts here are approximated as whitespace-split word counts, not a
real tokenizer's subword count. That's deliberate: chunk sizing only needs
to be roughly consistent across strategies, not billing-accurate, and it
keeps this module free of a tokenizer dependency.
"""

from __future__ import annotations

import re

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def approx_token_count(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def chunk_fixed_size(text: str, size_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Cuts on raw word boundaries at a fixed size, ignoring sentence or
    paragraph structure entirely. The naive baseline every other strategy
    is trying to improve on: fast and simple, but a cut can land mid-sentence."""
    words = text.split()
    if not words:
        return []
    step = max(size_tokens - overlap_tokens, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + size_tokens]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + size_tokens >= len(words):
            break
    return chunks


def chunk_sentence(text: str, target_tokens: int) -> list[str]:
    """Greedily packs whole sentences into a chunk until adding the next
    sentence would exceed target_tokens, then starts a new chunk. Never
    cuts a sentence in half, unlike chunk_fixed_size."""
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = approx_token_count(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_recursive(text: str, target_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Splits on paragraph breaks first, then falls back to sentences for
    any paragraph still too big, then to fixed-size words as a last resort
    for a single run-on sentence. Adjacent pieces are merged back up toward
    target_tokens so a chunk isn't just "one paragraph, whatever size that
    happens to be." This is the LangChain-style RecursiveCharacterTextSplitter
    pattern, and the default chunking strategy for the main 6-architecture
    comparison, since it's the most commonly reached-for strategy in
    practice and a reasonable middle ground between the crude fixed-size
    cut and the more expensive semantic strategy below."""

    def split_piece(piece: str) -> list[str]:
        if approx_token_count(piece) <= target_tokens:
            return [piece]
        sentences = _split_sentences(piece)
        if len(sentences) > 1:
            pieces: list[str] = []
            for sentence in sentences:
                pieces.extend(split_piece(sentence))
            return pieces
        return chunk_fixed_size(piece, target_tokens, overlap_tokens)

    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]
    if not paragraphs:
        return []

    atoms: list[str] = []
    for paragraph in paragraphs:
        atoms.extend(split_piece(paragraph))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for atom in atoms:
        atom_tokens = approx_token_count(atom)
        if current and current_tokens + atom_tokens > target_tokens:
            chunks.append(" ".join(current))
            if overlap_tokens > 0:
                overlap_words = " ".join(current).split()[-overlap_tokens:]
                current, current_tokens = [" ".join(overlap_words)], len(overlap_words)
            else:
                current, current_tokens = [], 0
        current.append(atom)
        current_tokens += atom_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_semantic(text: str, embedder, similarity_threshold: float = 0.55, min_sentences: int = 3) -> list[str]:
    """Embeds every sentence, then cuts a new chunk wherever the cosine
    similarity between consecutive sentences drops below
    similarity_threshold, the idea being that a topic shift shows up as a
    dip in how similar two neighboring sentences are. This is the most
    expensive strategy (one embedding call per sentence rather than per
    chunk) and the one every other strategy here is implicitly being
    checked against: does cutting on meaning actually beat cutting on a
    fixed size?  Method follows the breakpoint-based semantic chunking
    described in https://research.trychroma.com/evaluating-chunking.
    """
    import numpy as np

    sentences = _split_sentences(text)
    if len(sentences) <= min_sentences:
        return [" ".join(sentences)] if sentences else []

    embeddings = embedder.encode(sentences, normalize_embeddings=True)
    chunks: list[str] = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
        if similarity < similarity_threshold and len(current) >= min_sentences:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


STRATEGIES = {
    "fixed_128": lambda text, embedder=None: chunk_fixed_size(text, 128, overlap_tokens=16),
    "fixed_256": lambda text, embedder=None: chunk_fixed_size(text, 256, overlap_tokens=32),
    "fixed_512": lambda text, embedder=None: chunk_fixed_size(text, 512, overlap_tokens=64),
    "sentence": lambda text, embedder=None: chunk_sentence(text, 180),
    "recursive": lambda text, embedder=None: chunk_recursive(text, 220, overlap_tokens=30),
    "semantic": lambda text, embedder=None: chunk_semantic(text, embedder),
}

DEFAULT_STRATEGY = "recursive"
