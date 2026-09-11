"""Casts a wide dense-retrieval net (top-20), then reranks that shortlist
with a cross-encoder before keeping only the top-5 for generation. The bet:
a cross-encoder that reads the query and a chunk together in one forward
pass can judge relevance more precisely than cosine similarity between two
independently-computed embeddings, but is too slow to run over an entire
corpus, so it only ever sees a shortlist a cheaper method narrowed first.
"""

import time

from retrieval import dense_search, rerank
from techniques.common import ANSWER_SYSTEM_PROMPT, RunResult, generate_answer, llm_step, retrieval_step

NAME = "Reranked RAG"
DESCRIPTION = (
    "Dense search retrieves a wide top-20 shortlist, a cross-encoder reranks it, the "
    "top-5 by rerank score go to the generator."
)
RETRIEVE_K = 20
RERANK_K = 5
PROMPTS = {"generator": ANSWER_SYSTEM_PROMPT}

WHAT_IT_IS = (
    "A bi-encoder (what dense_search uses) embeds the query and every chunk *independently*, "
    "then compares the two vectors with cosine similarity — fast enough to run over an entire "
    "corpus, because every chunk's embedding is computed once, offline, before any query "
    "arrives. A cross-encoder instead feeds the query and one candidate chunk into the model "
    "*together*, so it can attend across the two texts directly rather than comparing two "
    "fixed summaries of them. That's more accurate, and also why it can't be the first-pass "
    "retriever: it has to run once per (query, chunk) pair, at query time, so scoring the whole "
    "corpus this way would be far too slow. Two-stage retrieve-then-rerank is the standard fix: "
    "let a fast bi-encoder narrow thousands of chunks down to a shortlist, then let a slower, "
    "more accurate cross-encoder re-order just that shortlist."
)
HOW_WE_IMPLEMENTED_IT = (
    "`dense_search()` returns the top 20 chunks by cosine similarity, the same call naive RAG "
    "makes with k=5 instead of k=20. `rerank()` scores each of those 20 with "
    "`cross-encoder/ms-marco-MiniLM-L-6-v2` (query and chunk text together, one forward pass "
    "per pair) and keeps the top 5 by that score. Those 5 go into the same shared "
    "`generate_answer()` prompt. No LLM call is spent on reranking — the cross-encoder is a "
    "small, dedicated model, not the generation model — so this costs one extra local model "
    "call, not one extra Ollama round trip."
)
WHEN_ITS_USEFUL = (
    "Reach for this when the corpus has a lot of chunks that are topically close but only one "
    "or two are actually precisely relevant, the situation where cosine similarity's coarser "
    "notion of 'close in embedding space' most often over- or under-ranks a chunk relative to "
    "what a closer read would show. It's a good fit whenever you can afford one extra local "
    "model call per query (the cross-encoder here runs on CPU in well under a second per "
    "shortlist), which is nearly always, since it needs no extra LLM call at all. It's a weaker "
    "fit if the first-pass retrieval's top-20 already excludes the actually-relevant chunk "
    "entirely, since a reranker can only reorder what it's shown, and it doesn't help a query "
    "that's failing because of how it's phrased rather than how chunks are ranked, which is "
    "what HyDE and decomposition address instead."
)
DIAGRAM = """flowchart LR
    Q["Query"] --> D["Dense search<br/>top-20"]
    D --> R["Cross-encoder rerank"]
    R --> K["Top-5 by rerank score"]
    K --> G["Generate answer"]
    G --> Ans["Predicted answer"]
"""


def run(query: dict, index) -> RunResult:
    started_at = time.perf_counter()
    steps = []

    retrieval_started = time.perf_counter()
    candidates = dense_search(index, query["question"], RETRIEVE_K)
    steps.append(retrieval_step("retriever", query["question"], candidates, time.perf_counter() - retrieval_started))

    rerank_started = time.perf_counter()
    reranked_chunks = rerank(query["question"], candidates, RERANK_K)
    steps.append(retrieval_step("reranker", query["question"], reranked_chunks, time.perf_counter() - rerank_started))

    call, answer, abstained = generate_answer(query["question"], reranked_chunks)
    steps.append(llm_step("generator", "generate answer", call))

    return RunResult(
        predicted_answer=answer,
        steps=steps,
        retrieved_chunks=reranked_chunks,
        abstained=abstained,
        wall_clock_seconds=time.perf_counter() - started_at,
        retrieval_rounds=1,
    )
