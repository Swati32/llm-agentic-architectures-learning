"""Runs two independent retrieval signals over the same query, dense
(semantic) and BM25 (lexical), and merges their rankings with reciprocal
rank fusion before generating an answer. The bet: dense and BM25 fail on
different queries (dense misses exact names/numbers it wasn't trained to
weight highly; BM25 misses a paraphrase that shares no words with the
source text), so a query that beats one of them often survives the other.
"""

import time

from retrieval import bm25_search, dense_search, reciprocal_rank_fusion
from techniques.common import ANSWER_SYSTEM_PROMPT, RunResult, generate_answer, llm_step, retrieval_step

NAME = "Hybrid RAG (Dense + BM25)"
DESCRIPTION = (
    "Retrieves with dense search and BM25 independently, fuses the two rankings with "
    "reciprocal rank fusion, keeps the top-5 fused chunks, generates an answer."
)
DENSE_K = 10
BM25_K = 10
FUSED_K = 5
PROMPTS = {"generator": ANSWER_SYSTEM_PROMPT}

WHAT_IT_IS = (
    "Dense embedding search is good at matching meaning: a query about 'the company's CEO "
    "stepping down' can find a passage that says 'resigned' without sharing a single content "
    "word. It's weaker on exact strings, an uncommon proper noun, a model number, a date, "
    "because those get compressed into the same embedding space as everything else and don't "
    "stand out the way they do to a keyword matcher. BM25 ([Robertson & Zaragoza]"
    "(https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf)) is the reverse: "
    "a classic term-frequency ranking function with no notion of meaning at all, so it nails an "
    "exact name or number but misses a paraphrase entirely. Hybrid search runs both and merges "
    "the two rankings, rather than picking one."
)
HOW_WE_IMPLEMENTED_IT = (
    "`dense_search()` and `bm25_search()` each independently rank the whole corpus and return "
    "their top 10. `reciprocal_rank_fusion()` merges the two ranked lists by *position*, not "
    "raw score, using the standard 1/(k + rank) formula from [Cormack et al., 2009]"
    "(https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — this sidesteps the problem "
    "that a cosine similarity and a BM25 score live on completely different, incomparable "
    "scales. The top 5 chunks by fused score go into the same shared `generate_answer()` prompt "
    "every architecture uses. One retrieval round, two retrieval calls inside it."
)
WHEN_ITS_USEFUL = (
    "Reach for this when queries are a mix of natural-language paraphrase and exact-term "
    "lookup, which most real corpora are: a news corpus has both 'who criticized the merger' "
    "(semantic) and 'Q3 2023 earnings' (lexical) style queries, often inside the same question. "
    "It costs one extra retrieval call over naive RAG and no extra LLM call, so it's close to "
    "free to try. It's a weaker fit when the corpus is narrow enough that dense search alone "
    "already covers the vocabulary well, or when what's actually failing is the query's framing "
    "rather than the retrieval mechanism, in which case rewriting the query (HyDE, "
    "decomposition) helps more than adding a second ranking signal."
)
DIAGRAM = """flowchart LR
    Q["Query"] --> DE["Dense search<br/>top-10"]
    Q --> BM["BM25 search<br/>top-10"]
    DE --> F["Reciprocal rank fusion"]
    BM --> F
    F --> K["Top-5 fused chunks"]
    K --> G["Generate answer"]
    G --> Ans["Predicted answer"]
"""


def run(query: dict, index) -> RunResult:
    started_at = time.perf_counter()
    steps = []

    retrieval_started = time.perf_counter()
    dense_chunks = dense_search(index, query["question"], DENSE_K)
    bm25_chunks = bm25_search(index, query["question"], BM25_K)
    fused_chunks = reciprocal_rank_fusion([dense_chunks, bm25_chunks], FUSED_K)
    retrieval_latency = time.perf_counter() - retrieval_started
    steps.append(retrieval_step("retriever", f"dense: {query['question']}", dense_chunks, retrieval_latency / 2))
    steps.append(retrieval_step("retriever", f"bm25: {query['question']}", bm25_chunks, retrieval_latency / 2))

    call, answer, abstained = generate_answer(query["question"], fused_chunks)
    steps.append(llm_step("generator", "generate answer", call))

    return RunResult(
        predicted_answer=answer,
        steps=steps,
        retrieved_chunks=fused_chunks,
        abstained=abstained,
        wall_clock_seconds=time.perf_counter() - started_at,
        retrieval_rounds=1,
    )
