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

WHAT_IT_IS = (
    "The assembly-line pattern: break a task into a fixed sequence of stages, each with one "
    "narrow responsibility, and run every stage in the same order every time. There's no "
    "planning step that adapts to what a stage finds; the shape of the pipeline (how many "
    "stages, what each one does) is decided once, in the code, not per question. This is the "
    "simplest way to get multiple specialized roles cooperating: each stage is easy to write, "
    "easy to test in isolation, and easy to reason about, because it only ever sees the fixed "
    "inputs its position in the pipeline gives it."
)
HOW_WE_IMPLEMENTED_IT = (
    "Four stages, always in this order: a Decomposer call turns the question into exactly 2 "
    "sub-questions upfront (regex-parsed from a numbered list; if parsing fails, both hops fall "
    "back to asking the original question). Then Hop 1 and Hop 2 each run the same fixed "
    "sequence: `search()` the sub-question, then a Reader call answers it from the retrieved "
    "passage. Neither hop ever looks at the other's output. Finally a Synthesizer call is shown "
    "the original question plus both sub-question/answer pairs and produces the final answer. "
    "Handoffs are always exactly 3 (Decomposer to Hop 1, Hop 1 to Hop 2, Hop 2 to Synthesizer) "
    "and the step count is always 4 model calls plus 2 tool calls, for every question, which is "
    "what makes this architecture's cost so predictable next to the adaptive ones."
)
WHEN_ITS_USEFUL = (
    "Best when you already know the task's shape and the sub-tasks don't depend on each other's "
    "results, for example comparison questions, where 'how tall is A' and 'how tall is B' can be "
    "looked up in either order without needing the other's answer first. You get fixed, "
    "predictable latency and token cost (useful for capacity planning and pricing), and each "
    "stage is independently testable, which matters once you have more than a couple of stages "
    "to maintain. It's the wrong choice once a later stage genuinely needs an earlier stage's "
    "output to even know what to ask, our bridge-question results are the direct evidence: the "
    "Decomposer has to guess hop 2's wording before hop 1 has run, and often guesses wrong."
)
DIAGRAM = """flowchart LR
    Q["Question"] --> D["Decomposer<br/>writes 2 sub-questions"]
    D --> H1["Hop 1<br/>search + read"]
    H1 --> H2["Hop 2<br/>search + read"]
    H2 --> S["Synthesizer"]
    S --> Ans["Predicted answer"]
"""


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
