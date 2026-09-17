"""The RAG shape: one retrieval, then one generation, no loop deciding
whether to search again, no second role checking the answer. Adds exactly
one capability on top of the plain LLM shape (a lookup) and nothing else,
using the identical `search()` tool the single-agent and multi-agent shapes
also use, so any difference between this shape and those two comes from
whether the system can *decide* to search again, not from a different or
better retrieval mechanism.

Grounded in retrieval-augmented generation generally
(https://arxiv.org/abs/2005.11401): give the model a search step before it
answers, instead of relying only on what it memorized.
"""

import json
import time

from llm_client import call_model
from retrieval import search
from techniques.common import RunResult, llm_step, tool_step

NAME = "RAG (single-shot retrieval)"
DESCRIPTION = (
    "Retrieve once with the shared search() tool, then generate one answer from "
    "whatever it returned. No loop, no second role, no chance to notice a first "
    "search wasn't enough and search again."
)

READER_SYSTEM_PROMPT = (
    "Answer the question using only the passage given. Give the shortest correct "
    "answer (a name, date, yes or no, etc.), not a full sentence. If the passage "
    "does not contain the answer, say so.\n"
    "Respond with exactly:\nAnswer: <answer>"
)
PROMPTS = {"model": READER_SYSTEM_PROMPT}

WHAT_IT_IS = (
    "Retrieval-augmented generation in its simplest, non-adaptive form: search once, hand "
    "whatever comes back to the model, generate one answer. The model never sees the question "
    "without also seeing retrieved evidence, and it never gets a second chance to search "
    "differently if that evidence turns out to be wrong or incomplete."
)
HOW_WE_IMPLEMENTED_IT = (
    "`run()` calls the exact same `search(corpus, query)` tool the single-agent and multi-agent "
    "shapes call, with the raw question as the query and k=1 (the tool's only mode in this "
    "experiment, see `retrieval.py`), then makes one generation call with that single retrieved "
    "passage. With 2 gold paragraphs needed per HotpotQA question and only 1 ever retrieved, "
    "this shape structurally cannot see both pieces of evidence a bridge or comparison question "
    "needs, on purpose: that ceiling, not a weak retriever, is the point of comparing it against "
    "the single-agent shape, which uses the identical tool but can call it more than once. "
    "There is no loop and no second role, so `handoffs` is 0 and `early_terminated` is always "
    "`False`: one search, one generation, always completes."
)
WHEN_ITS_USEFUL = (
    "Reach for this when one lookup is genuinely enough, a single document or passage answers "
    "the question outright, and the extra cost and complexity of a loop or a second role would "
    "buy nothing. It's a weak fit for a task that needs more than one fact stitched together, "
    "like HotpotQA's bridge and comparison questions, which is exactly what this experiment "
    "uses it to demonstrate: this shape's gap versus the single-agent shape *is* the measured "
    "cost of not being able to decide to search again."
)
DIAGRAM = """flowchart LR
    Q["Question"] --> T["search() tool<br/>(one call)"]
    T --> M["Model + retrieved passage"]
    M --> Ans["Predicted answer"]
"""


def _answer(question: str, passage_title: str, passage_text: str):
    messages = [
        {"role": "system", "content": READER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage ({passage_title}): {passage_text}\n\nQuestion: {question}"},
    ]
    call = call_model(messages)
    content = call.content or ""
    if content.lower().startswith("answer:"):
        answer = content.split(":", 1)[1].strip()
    else:
        answer = content.strip() or None
    return call, answer


def run(example: dict, corpus) -> RunResult:
    question = example["question"]
    steps = []
    started_at = time.perf_counter()

    tool_started_at = time.perf_counter()
    try:
        result = search(corpus, question)
        tool_error = None
    except Exception as exc:  # noqa: BLE001 - a retrieval failure still ends this single-shot run cleanly
        result, tool_error = None, str(exc)
    tool_latency = time.perf_counter() - tool_started_at
    steps.append(tool_step("model", question, result, tool_latency, tool_error))

    predicted_answer = None
    if result is not None:
        call, predicted_answer = _answer(question, result["title"], result["text"])
        steps.append(llm_step("model", "answer from retrieved passage", call))

    return RunResult(
        predicted_answer=predicted_answer,
        steps=steps,
        handoffs=0,
        early_terminated=False,
        state_overhead_bytes=len(json.dumps({"question": question}).encode("utf-8")),
        wall_clock_seconds=time.perf_counter() - started_at,
    )
