"""Splits the document into paragraphs, embeds them, and keeps only the
top-k paragraphs most semantically similar to the question. A single
dense-search pass, no fusion or reranking (see the separate
rag-architectures experiment for a deep comparison of retrieval
architectures); the point here is narrower: does choosing what to keep
by relevance, instead of keeping everything or keeping only what's
recent, change what a bounded context can still answer correctly.
"""

import time

from retrieval import select_relevant_paragraphs
from techniques.common import ANSWER_SYSTEM_PROMPT, RunResult, generate_answer, llm_step

NAME = "Retrieval Selection"
DESCRIPTION = "Embeds every paragraph, keeps only the top-k most similar to the question."
TOP_K_PARAGRAPHS = 3
PROMPTS = {"generator": ANSWER_SYSTEM_PROMPT}

WHAT_IT_IS = (
    "Splits the document into its paragraphs, embeds each one and the question with the same "
    f"model, and keeps only the {TOP_K_PARAGRAPHS} paragraphs whose embedding is most similar to "
    "the question's, restored to their original order. This is the leanest possible form of "
    "retrieval, one dense search, no reranking or fusion, applied here to *context management* "
    "rather than to answering from an external corpus: the 'corpus' is just this one document's "
    "own paragraphs. Same underlying idea as [Lewis et al., 2020](https://arxiv.org/abs/2005.11401)'s "
    "retrieval-augmented generation, at document scale instead of corpus scale."
)
HOW_WE_IMPLEMENTED_IT = (
    "`select_relevant_paragraphs()` (see [retrieval.py](../retrieval.py)) embeds every paragraph "
    "with `all-MiniLM-L6-v2`, ranks them by cosine similarity to the question, and keeps the top "
    f"{TOP_K_PARAGRAPHS}, restored to their original document order before being joined and passed "
    "into the shared `generate_answer()` prompt (see [techniques/common.py](common.py)). One "
    "embedding pass, one LLM call."
)
WHEN_ITS_USEFUL = (
    "The right choice whenever the question itself is a good enough signal for what's relevant, "
    "and the content that answers it resembles the question well semantically, exactly the "
    "condition needle-in-haystack questions satisfy by construction, since the needle paragraph "
    "is literally the source the question was written from. It's a weaker fit when a question "
    "needs several scattered pieces combined (this keeps only the top-k by similarity to the "
    "question alone, it doesn't reason about what else might be needed), or when nothing in the "
    "document actually resembles the question's wording even though the answer is in there "
    "somewhere."
)
DIAGRAM = f"""flowchart LR
    D["Document"] --> P["Split into paragraphs"]
    P --> E["Embed paragraphs + question"]
    E --> S["Keep top-{TOP_K_PARAGRAPHS} by similarity"]
    S --> G["Generate answer<br/>from selected paragraphs"]
    G --> Ans["Predicted answer"]
"""


def run(document: str, question: str) -> RunResult:
    started_at = time.perf_counter()
    selected_text = "\n\n".join(select_relevant_paragraphs(document, question, TOP_K_PARAGRAPHS))
    call, answer = generate_answer(question, selected_text)
    steps = [llm_step("generator", "generate answer from retrieved paragraphs", call)]
    return RunResult(
        predicted_answer=answer,
        steps=steps,
        context_sent_to_generator=selected_text,
        wall_clock_seconds=time.perf_counter() - started_at,
    )
