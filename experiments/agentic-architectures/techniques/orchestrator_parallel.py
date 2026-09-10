"""Same fixed, upfront decomposition as sequential_pipeline.py, but the two
Worker lookups are dispatched concurrently instead of one after another —
fan-out to two workers, fan-in to one Synthesizer. Tests whether
parallelizing sub-questions actually saves wall-clock time, and exposes the
same blind-decomposition weakness as the fixed pipeline, sharper: a worker
running in parallel cannot ever see the other worker's answer, so a truly
dependent second hop (needing an entity the first hop hasn't found yet) is
structurally impossible here, not just poorly guessed.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

from retrieval import search
from techniques.common import (
    DECOMPOSER_SYSTEM_PROMPT,
    READER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    RunResult,
    Step,
    decompose,
    llm_step,
    read_answer,
    synthesize,
    tool_step,
)

NAME = "Orchestrator (parallel dispatch)"
DESCRIPTION = (
    "Decomposer writes both sub-questions upfront, then both are retrieved and "
    "answered concurrently by independent Workers, then a Synthesizer combines "
    "them. Trades any chance of hop 2 using hop 1's answer for lower wall-clock "
    "time, when the serving backend actually parallelizes."
)
PROMPTS = {
    "decomposer": DECOMPOSER_SYSTEM_PROMPT,
    "worker (reader)": READER_SYSTEM_PROMPT,
    "synthesizer": SYNTHESIZER_SYSTEM_PROMPT,
}

WHAT_IT_IS = (
    "The same orchestrator-workers pattern as the sequential-dispatch version, but with a "
    "different dispatch policy: instead of running Workers one at a time and letting each one "
    "see the last one's result, all the Workers are launched together and the Orchestrator "
    "waits for all of them before moving on. This is the fan-out/fan-in shape: one decision "
    "upstream spawns several independent branches of work, and a downstream step reassembles "
    "them. It trades the *possibility* of one Worker informing another for the *possibility* of "
    "finishing faster, and it only delivers the second part if whatever is actually serving the "
    "model can run those Workers' calls concurrently rather than queueing them."
)
HOW_WE_IMPLEMENTED_IT = (
    "Identical decomposition step to the fixed Sequential Pipeline: one Decomposer call produces "
    "2 sub-questions upfront. Then both are dispatched at once with Python's `ThreadPoolExecutor`, "
    "each running its own `search()` and Reader call in its own thread, and we wait for both "
    "before continuing. Results are collected back in the original order regardless of which "
    "thread finished first, then handed to the same Synthesizer used elsewhere. Handoffs count "
    "as 4 (Decomposer to each of 2 Workers, each Worker back to the Synthesizer), one more than "
    "the fixed pipeline's 3, because a fan-out/fan-in shape has more edges than a straight chain "
    "even though it does the same amount of work. The wall-clock benefit is only real if the "
    "backend serving the model actually processes concurrent requests in parallel."
)
WHEN_ITS_USEFUL = (
    "Use this when your sub-tasks are genuinely independent (nothing downstream needs to see "
    "another sub-task's result before it can run) and your serving infrastructure can actually "
    "run concurrent model calls, batched GPU serving, multiple replicas behind a load balancer, "
    "or a hosted API with real concurrency. In that setting it buys you latency for free: same "
    "answers, same cost, less wall-clock time. It buys you nothing, and can quietly produce "
    "wrong answers, when a later sub-task actually depends on an earlier one's result: a worker "
    "running concurrently structurally cannot see another worker's output, so a genuinely "
    "chained question can never be answered correctly this way, no matter how good the model is."
)
DIAGRAM = """flowchart LR
    Q["Question"] --> D["Decomposer<br/>writes 2 sub-questions"]
    D --> W1["Worker 1<br/>search + read"]
    D --> W2["Worker 2<br/>search + read"]
    W1 --> S["Synthesizer"]
    W2 --> S
    S --> Ans["Predicted answer"]
"""


def _run_worker(worker_index: int, sub_question: str, corpus) -> tuple[list[Step], str | None]:
    worker_steps = []
    tool_started_at = time.perf_counter()
    try:
        result = search(corpus, sub_question)
        tool_error = None
    except Exception as exc:  # noqa: BLE001 - a retrieval failure ends this worker, not the run
        result, tool_error = None, str(exc)
    tool_latency = time.perf_counter() - tool_started_at
    worker_steps.append(tool_step(f"worker {worker_index}", sub_question, result, tool_latency, tool_error))

    if result is None:
        return worker_steps, None
    reader_call, sub_answer = read_answer(sub_question, result["title"], result["text"])
    worker_steps.append(llm_step(f"worker {worker_index}", sub_question, reader_call))
    return worker_steps, sub_answer


def run(example: dict, corpus) -> RunResult:
    question = example["question"]
    steps = []
    started_at = time.perf_counter()

    decomposer_call, sub_questions = decompose(question)
    steps.append(llm_step("decomposer", "decompose question", decomposer_call))

    with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
        futures = [
            executor.submit(_run_worker, index, sub_question, corpus)
            for index, sub_question in enumerate(sub_questions, start=1)
        ]
        worker_outputs = [future.result() for future in futures]

    sub_qas = []
    for sub_question, (worker_steps, sub_answer) in zip(sub_questions, worker_outputs):
        steps.extend(worker_steps)
        sub_qas.append((sub_question, sub_answer))

    state_overhead_bytes = len(json.dumps({"question": question, "sub_qas": sub_qas}).encode("utf-8"))

    synthesizer_call, predicted_answer = synthesize(question, sub_qas)
    steps.append(llm_step("synthesizer", "combine worker answers", synthesizer_call))

    return RunResult(
        predicted_answer=predicted_answer,
        steps=steps,
        handoffs=2 * len(sub_questions),  # decomposer->worker and worker->synthesizer, per worker
        early_terminated=synthesizer_call.error is not None or predicted_answer is None,
        state_overhead_bytes=state_overhead_bytes,
        wall_clock_seconds=time.perf_counter() - started_at,
    )
