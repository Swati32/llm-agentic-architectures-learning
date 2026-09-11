"""Breaks the question into 2-3 simpler, self-contained sub-questions,
retrieves separately for each, merges every sub-question's chunks into one
context, and generates one final answer from the union. The bet: a single
embedding of a multi-part question is a compromise vector that partially
matches several distinct topics at once and therefore matches any one of
them worse than a query embedding aimed squarely at just that topic would.

Related to the decomposition step used in this repo's agentic-architectures
experiment, but here every sub-question is searched in the same round
rather than fed forward sequentially into the next sub-question, since
MultiHop-RAG's question types (comparison, inference, temporal) don't
require one hop's answer to phrase the next hop's query the way
HotpotQA's bridge questions do.
"""

import time

from retrieval import dense_search
from techniques.common import ANSWER_SYSTEM_PROMPT, DECOMPOSE_SYSTEM_PROMPT, RunResult, decompose_query, generate_answer, llm_step, retrieval_step

NAME = "Query Decomposition RAG"
DESCRIPTION = (
    "Breaks the query into 2-3 sub-questions, retrieves top chunks per sub-question, "
    "merges them, generates one answer from the union."
)
CHUNKS_PER_SUBQUERY = 3
PROMPTS = {
    "decomposer": DECOMPOSE_SYSTEM_PROMPT,
    "generator": ANSWER_SYSTEM_PROMPT,
}

WHAT_IT_IS = (
    "A comparison question like 'which of X and Y was founded first' is really two lookups "
    "wearing one sentence: 'when was X founded' and 'when was Y founded'. Embedded as a single "
    "vector, the question is a blend of both topics, and a chunk about X alone or Y alone can "
    "each match that blended vector worse than they'd match a query aimed at just one of them. "
    "Query decomposition asks the model to split a question into that many separate, "
    "self-contained sub-questions first, retrieves for each one independently, and only "
    "combines everything at the generation step."
)
HOW_WE_IMPLEMENTED_IT = (
    "`decompose_query()` asks the model for 2 to 3 sub-questions, each required to stand alone "
    "(no 'it' or 'that company' referring back to something not yet named, since there's no "
    "sequential hand-off between sub-questions here to resolve a pronoun against). Each "
    "sub-question gets its own `dense_search()` call, top 3 chunks; every sub-question's chunks "
    "are concatenated and de-duplicated by chunk id into one merged context, capped implicitly "
    "by however many distinct chunks 2-3 searches of 3 chunks each actually turn up (at most 9, "
    "usually fewer once overlap is removed). That merged context goes into the same shared "
    "`generate_answer()` prompt every architecture uses. If decomposition fails to parse into at "
    "least one sub-question, the run falls back to searching the original question once."
)
WHEN_ITS_USEFUL = (
    "Reach for this on genuinely multi-part questions, comparisons, questions that need facts "
    "from more than one distinct source, where a single retrieval pass has to compromise "
    "between topics. It costs one extra LLM call up front (the decomposition itself) plus "
    "however many extra retrieval calls there are sub-questions, but no extra generation call, "
    "since everything is still answered in one final pass. It's a weaker fit on a genuinely "
    "single-fact question: decomposing something that didn't need decomposing just adds an LLM "
    "call and a chance of the 'sub-questions' drifting away from what was actually asked, and "
    "it doesn't help when a query's problem is its phrasing rather than its being multi-part, "
    "which is what HyDE targets instead."
)
DIAGRAM = """flowchart LR
    Q["Query"] --> DC["LLM: decompose into<br/>2-3 sub-questions"]
    DC --> S1["Search sub-q 1<br/>top-3"]
    DC --> S2["Search sub-q 2<br/>top-3"]
    DC --> S3["Search sub-q 3<br/>top-3"]
    S1 --> M["Merge + dedupe chunks"]
    S2 --> M
    S3 --> M
    M --> G["Generate answer<br/>from merged context"]
    G --> Ans["Predicted answer"]
"""


def run(query: dict, index) -> RunResult:
    started_at = time.perf_counter()
    steps = []

    decompose_call, sub_questions = decompose_query(query["question"])
    steps.append(llm_step("decomposer", "decompose query", decompose_call))

    merged_chunks = []
    seen_ids = set()
    for sub_question in sub_questions:
        retrieval_started = time.perf_counter()
        chunks = dense_search(index, sub_question, CHUNKS_PER_SUBQUERY)
        steps.append(retrieval_step("sub_query_retriever", sub_question, chunks, time.perf_counter() - retrieval_started))
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                merged_chunks.append(chunk)

    call, answer, abstained = generate_answer(query["question"], merged_chunks)
    steps.append(llm_step("generator", "generate answer", call))

    return RunResult(
        predicted_answer=answer,
        steps=steps,
        retrieved_chunks=merged_chunks,
        abstained=abstained,
        wall_clock_seconds=time.perf_counter() - started_at,
        retrieval_rounds=1,  # all sub-queries are searched in the same round, not sequentially
    )
