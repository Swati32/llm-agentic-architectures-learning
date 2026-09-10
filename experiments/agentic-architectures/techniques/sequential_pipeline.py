"""Fixed sequential pipeline: Decomposer -> Hop 1 -> Hop 2 -> Synthesizer,
always in that order, decided entirely upfront. Once the decomposer has
written its two sub-questions, nothing learned during hop 1 changes hop 2's
query — this is the "assembly line" style of multi-agent system, as
opposed to the adaptive orchestrator in orchestrator_sequential.py.

The expected failure mode is bridge questions: the decomposer has to guess
hop 2's wording (e.g. "What team did that person play for?") without
knowing the entity hop 1 will actually surface, since it never sees hop 1's
answer before writing hop 2.
"""

import json
import time

from retrieval import search
from techniques.common import (
    DECOMPOSER_SYSTEM_PROMPT,
    READER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    RunResult,
    decompose,
    llm_step,
    read_answer,
    synthesize,
    tool_step,
)

NAME = "Sequential Pipeline (fixed)"
DESCRIPTION = (
    "Decomposer writes both sub-questions upfront, then Hop 1 and Hop 2 each "
    "retrieve and answer in fixed order, then a Synthesizer combines them. No "
    "stage ever revisits an earlier one's output."
)
PROMPTS = {
    "decomposer": DECOMPOSER_SYSTEM_PROMPT,
    "reader (hop 1 and hop 2)": READER_SYSTEM_PROMPT,
    "synthesizer": SYNTHESIZER_SYSTEM_PROMPT,
}


def run(example: dict, corpus) -> RunResult:
    question = example["question"]
    steps = []
    started_at = time.perf_counter()

    decomposer_call, sub_questions = decompose(question)
    steps.append(llm_step("decomposer", "decompose question", decomposer_call))

    sub_qas = []
    for hop_index, sub_question in enumerate(sub_questions, start=1):
        tool_started_at = time.perf_counter()
        try:
            result = search(corpus, sub_question)
            tool_error = None
        except Exception as exc:  # noqa: BLE001 - a retrieval failure ends this hop, not the run
            result, tool_error = None, str(exc)
        tool_latency = time.perf_counter() - tool_started_at
        steps.append(tool_step(f"hop {hop_index}", sub_question, result, tool_latency, tool_error))

        if result is None:
            sub_qas.append((sub_question, None))
            continue
        reader_call, sub_answer = read_answer(sub_question, result["title"], result["text"])
        steps.append(llm_step(f"hop {hop_index}", sub_question, reader_call))
        sub_qas.append((sub_question, sub_answer))

    state_overhead_bytes = len(json.dumps({"question": question, "sub_qas": sub_qas}).encode("utf-8"))

    synthesizer_call, predicted_answer = synthesize(question, sub_qas)
    steps.append(llm_step("synthesizer", "combine hop answers", synthesizer_call))

    return RunResult(
        predicted_answer=predicted_answer,
        steps=steps,
        handoffs=3,  # decomposer -> hop1 -> hop2 -> synthesizer, always fixed
        early_terminated=synthesizer_call.error is not None or predicted_answer is None,
        state_overhead_bytes=state_overhead_bytes,
        wall_clock_seconds=time.perf_counter() - started_at,
    )
