"""Baseline: embed the raw query, take the top-k chunks by cosine
similarity, stuff them into the prompt, generate an answer. No rewriting,
no reranking, no second retrieval pass. This is what "RAG" means if you've
only ever seen one diagram of it, and the floor every other architecture
here is measured against: does adding structure (a second retrieval
signal, a rewrite step, a grading step) actually buy anything over this?

Grounded in the original RAG formulation, Lewis et al., 2020
(https://arxiv.org/abs/2005.11401): retrieve top-k passages by dense
similarity, condition generation on them.
"""

import time

from retrieval import dense_search
from techniques.common import ANSWER_SYSTEM_PROMPT, RunResult, generate_answer, llm_step, retrieval_step

NAME = "Naive RAG"
DESCRIPTION = (
    "Embed the query, take the top-5 chunks by cosine similarity, generate an answer "
    "from them directly. One retrieval call, one generation call. The baseline."
)
TOP_K = 5
PROMPTS = {"generator": ANSWER_SYSTEM_PROMPT}

WHAT_IT_IS = (
    "The retrieval step every RAG system starts from: embed the query into the same vector "
    "space the corpus was indexed in, rank every chunk by cosine similarity to that one vector, "
    "and keep the top k. There's no query rewriting, no second retrieval signal, no relevance "
    "check on what comes back — whatever the top-k similarity search returns is what the "
    "generator sees. From [Lewis et al., 2020](https://arxiv.org/abs/2005.11401), the paper that "
    "coined the term 'retrieval-augmented generation': condition a generator on retrieved "
    "passages instead of relying only on what it memorized during training."
)
HOW_WE_IMPLEMENTED_IT = (
    "`dense_search()` embeds the query with `all-MiniLM-L6-v2` and returns the 5 highest-cosine "
    "chunks from the shared corpus index (see [retrieval.py](../retrieval.py)). Those 5 chunks "
    "are formatted as numbered passages and passed, verbatim, into the shared "
    "`generate_answer()` prompt every architecture in this experiment uses (see "
    "[techniques/common.py](common.py)), which is told to answer only from the given passages "
    "and to say 'Insufficient information.' if they don't support an answer. One retrieval call, "
    "one generation call, nothing else."
)
WHEN_ITS_USEFUL = (
    "Reach for this first, always. It's the cheapest architecture here by a wide margin (one "
    "embedding call, one LLM call, no reranking model, no extra retrieval round), and it's a "
    "real baseline, not a strawman: on a corpus where the right chunk usually resembles the "
    "query well and a single well-formed lookup is enough, the more elaborate architectures buy "
    "little over this. It's a weaker fit once the query's wording is a poor semantic match for "
    "how the answer is actually phrased in the corpus (HyDE's whole reason to exist), once a "
    "single retrieval pass genuinely can't cover a multi-part question (query decomposition), or "
    "once getting the ranking right matters more than getting it fast (reranking)."
)
DIAGRAM = """flowchart LR
    Q["Query"] --> E["Embed query"]
    E --> D["Dense search<br/>top-5 chunks"]
    D --> G["Generate answer<br/>from 5 passages"]
    G --> Ans["Predicted answer"]
"""


def run(query: dict, index) -> RunResult:
    started_at = time.perf_counter()
    steps = []

    retrieval_started = time.perf_counter()
    chunks = dense_search(index, query["question"], TOP_K)
    steps.append(retrieval_step("retriever", query["question"], chunks, time.perf_counter() - retrieval_started))

    call, answer, abstained = generate_answer(query["question"], chunks)
    steps.append(llm_step("generator", "generate answer", call))

    return RunResult(
        predicted_answer=answer,
        steps=steps,
        retrieved_chunks=chunks,
        abstained=abstained,
        wall_clock_seconds=time.perf_counter() - started_at,
        retrieval_rounds=1,
    )
