"""Processes the document as a stream of fixed-size chunks, folding each
one into a running summary as it goes, then answers from the running
summary plus the last raw chunk. Mirrors what a real long-running agent's
context compaction does (see the Claude API's own compaction feature):
older content is never dropped outright, it's compressed, so the gist of
everything survives, but any specific fact the summarizer chose not to
keep is gone for good.
"""

import time

from techniques.common import ANSWER_SYSTEM_PROMPT, RunResult, SUMMARIZE_SYSTEM_PROMPT, generate_answer, llm_step, summarize_chunk

NAME = "Hierarchical Summarization"
DESCRIPTION = "Folds the document into a running summary chunk by chunk, keeps the last raw chunk verbatim."
CHUNK_TOKENS = 1200
PROMPTS = {"generator": ANSWER_SYSTEM_PROMPT, "summarizer": SUMMARIZE_SYSTEM_PROMPT}

WHAT_IT_IS = (
    f"Splits the document into ~{CHUNK_TOKENS}-word chunks and processes them in order, folding "
    "each one into a running summary before moving to the next, the same way a real long-running "
    "agent's context gets compacted incrementally rather than all at once. The final answer is "
    "generated from the running summary of every earlier chunk plus the last chunk kept verbatim. "
    "Grounded in the recurrent-memory line of long-context research, e.g. the [Compressive "
    "Transformer](https://arxiv.org/abs/1911.05507) and [Recurrent Memory Transformer]"
    "(https://arxiv.org/abs/2207.06881): compress what's aged out of the immediate window instead "
    "of discarding it outright."
)
HOW_WE_IMPLEMENTED_IT = (
    "`summarize_chunk()` (see [techniques/common.py](common.py)) makes one LLM call per chunk "
    "boundary, each time folding the new chunk into the running summary so far. A document under "
    f"{CHUNK_TOKENS} words needs no summarization call at all, since it's already one chunk. The "
    "final `generate_answer()` call sees the accumulated summary plus the last raw chunk, never "
    "the original text of any earlier chunk."
)
WHEN_ITS_USEFUL = (
    "Fits a long-running stream where the gist of older content still matters but keeping it "
    "verbatim doesn't (a long conversation, an agent's accumulated tool output) better than a "
    "sliding window, which drops older content's meaning entirely rather than compressing it. It "
    "costs one extra LLM call per chunk boundary over full context or a plain window, and it "
    "fails softly rather than not at all: whatever specific fact the summarizer judged "
    "unimportant enough to compress away is gone, even though something about that chunk "
    "survives."
)
DIAGRAM = """flowchart LR
    D["Document"] --> C["Split into chunks"]
    C --> S1["Earlier chunks"] --> Sum["Fold into<br/>running summary"]
    Sum --> G["Generate answer from<br/>running summary + last chunk"]
    C --> SN["Last chunk<br/>(kept verbatim)"] --> G
    G --> Ans["Predicted answer"]
"""


def _chunk_words(document: str, chunk_tokens: int) -> list[str]:
    words = document.split()
    return [" ".join(words[i : i + chunk_tokens]) for i in range(0, len(words), chunk_tokens)]


def run(document: str, question: str) -> RunResult:
    started_at = time.perf_counter()
    chunks = _chunk_words(document, CHUNK_TOKENS)
    steps = []

    running_summary = None
    for chunk in chunks[:-1]:
        call, running_summary = summarize_chunk(chunk, running_summary)
        steps.append(llm_step("summarizer", "fold chunk into running summary", call))

    last_chunk = chunks[-1] if chunks else ""
    context_text = (
        f"Summary of earlier text:\n{running_summary}\n\nMost recent text:\n{last_chunk}"
        if running_summary
        else last_chunk
    )

    call, answer = generate_answer(question, context_text)
    steps.append(llm_step("generator", "generate answer from summary + last chunk", call))

    return RunResult(
        predicted_answer=answer,
        steps=steps,
        context_sent_to_generator=context_text,
        wall_clock_seconds=time.perf_counter() - started_at,
    )
