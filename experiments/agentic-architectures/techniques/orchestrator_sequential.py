"""Adaptive orchestrator: a Planner agent dispatches one Worker lookup at a
time, sees that worker's answer, and only then decides the next
sub-question — so unlike sequential_pipeline.py, hop 2's query can use a
name or fact hop 1 just surfaced. This is the direct test of whether
adaptive planning actually beats a fixed upfront plan on bridge questions.

Grounded in the orchestrator-workers pattern
(https://www.anthropic.com/engineering/building-effective-agents) and
adaptive decomposition that re-plans after each step, e.g. ADaPT
(https://arxiv.org/abs/2311.05772).
"""

import json
import time

from retrieval import search
from techniques.common import (
    READER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    RunResult,
    llm_step,
    read_answer,
    synthesize,
    tool_step,
)
from llm_client import call_model

NAME = "Orchestrator (sequential dispatch)"
DESCRIPTION = (
    "A Planner dispatches one Worker lookup at a time and sees the result before "
    "deciding the next sub-question, so later queries can use facts earlier ones "
    "surfaced. Stops adaptively once the Planner judges it has enough."
)
MAX_ROUNDS = 3

PLANNER_SYSTEM_PROMPT = (
    "You are coordinating research to answer a question, one lookup at a time. "
    "You will be shown what has been found so far. Decide the single next "
    "sub-question needed to make progress — you may use any name or fact "
    "already found — or, if you now have enough information to answer the "
    "original question, say so.\n"
    "Respond with exactly one line:\nNext: <sub-question>\nor\nNext: DONE"
)
PROMPTS = {
    "planner": PLANNER_SYSTEM_PROMPT,
    "worker (reader)": READER_SYSTEM_PROMPT,
    "synthesizer": SYNTHESIZER_SYSTEM_PROMPT,
}


def _format_findings(sub_qas: list[tuple[str, str | None]]) -> str:
    if not sub_qas:
        return "Nothing found yet."
    return "\n".join(f"- {sub_q} -> {sub_a}" for sub_q, sub_a in sub_qas)


def _parse_next(content: str | None) -> str | None:
    if content is None:
        return None
    for line in content.splitlines():
        if line.strip().lower().startswith("next:"):
            return line.split(":", 1)[1].strip()
    return None


def run(example: dict, corpus) -> RunResult:
    question = example["question"]
    steps = []
    sub_qas: list[tuple[str, str | None]] = []
    handoffs = 0
    used_up_budget = True
    started_at = time.perf_counter()

    for round_index in range(1, MAX_ROUNDS + 1):
        planner_messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original question: {question}\n\nFound so far:\n{_format_findings(sub_qas)}"},
        ]
        planner_call = call_model(planner_messages)
        steps.append(llm_step("planner", f"round {round_index}", planner_call))
        next_step = _parse_next(planner_call.content)

        if next_step is None or next_step.strip().upper() == "DONE":
            used_up_budget = False
            break

        handoffs += 1  # planner -> worker
        tool_started_at = time.perf_counter()
        try:
            result = search(corpus, next_step)
            tool_error = None
        except Exception as exc:  # noqa: BLE001 - a retrieval failure ends this round, not the run
            result, tool_error = None, str(exc)
        tool_latency = time.perf_counter() - tool_started_at
        steps.append(tool_step("worker", next_step, result, tool_latency, tool_error))

        if result is None:
            sub_qas.append((next_step, None))
        else:
            reader_call, sub_answer = read_answer(next_step, result["title"], result["text"])
            steps.append(llm_step("worker", next_step, reader_call))
            sub_qas.append((next_step, sub_answer))
        handoffs += 1  # worker -> planner

    handoffs += 1  # planner -> synthesizer
    state_overhead_bytes = len(json.dumps({"question": question, "sub_qas": sub_qas}).encode("utf-8"))

    synthesizer_call, predicted_answer = synthesize(question, sub_qas)
    steps.append(llm_step("synthesizer", "combine findings", synthesizer_call))

    return RunResult(
        predicted_answer=predicted_answer,
        steps=steps,
        handoffs=handoffs,
        early_terminated=used_up_budget or synthesizer_call.error is not None or predicted_answer is None,
        state_overhead_bytes=state_overhead_bytes,
        wall_clock_seconds=time.perf_counter() - started_at,
    )
