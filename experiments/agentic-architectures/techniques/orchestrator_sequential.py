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

WHAT_IT_IS = (
    "The orchestrator-workers pattern (see [Anthropic's building effective agents]"
    "(https://www.anthropic.com/engineering/building-effective-agents)), made adaptive: a "
    "central Planner role decides what work is needed next, delegates one piece to a Worker, "
    "and only after seeing that Worker's result does it decide the next piece. This is different "
    "from deciding the whole plan upfront: the Planner's second decision is informed by the "
    "first decision's outcome, so it can, in principle, reuse a name or fact the first lookup "
    "surfaced when it wasn't available at the start. That adaptivity is also its risk: the "
    "Planner has to decide for itself when it has enough, there's no fixed number of stages "
    "telling it when to stop."
)
HOW_WE_IMPLEMENTED_IT = (
    "A loop bounded to 3 rounds. Each round, the Planner is shown the original question and "
    "every sub-question/answer pair found so far, and must respond with either `Next: "
    "<sub-question>` or `Next: DONE`. If it says `DONE` (or its output doesn't parse), the loop "
    "ends and we move to the Synthesizer. Otherwise, a Worker runs `search()` on that "
    "sub-question and a Reader call answers it from the retrieved passage, and the result is "
    "appended to the shared findings before looping back to the Planner. Each round that "
    "dispatches a Worker counts 2 handoffs (Planner to Worker, Worker back to Planner); the "
    "final move to the Synthesizer is a third kind of handoff. If the loop exhausts all 3 rounds "
    "without the Planner ever saying `DONE`, the run is flagged `early_terminated`, which is "
    "exactly what happened on all 40 of our runs (see the Comparison tab)."
)
WHEN_ITS_USEFUL = (
    "This is the right shape for tasks with a genuine dependency chain, where you don't know "
    "the second lookup's exact wording until the first one resolves, and you don't know upfront "
    "how many lookups the question will actually need. It's a poor fit when you can't put a "
    "reliable bound on the model's own willingness to stop: budget for the step cap to bind "
    "every time, not occasionally, unless you've specifically verified your model reliably "
    "self-terminates on your task. It also costs more per question than a fixed pipeline (more "
    "model calls, more tokens, and latency that varies by question), so it's worth the adaptivity "
    "only when the task actually needs it."
)
DIAGRAM = """flowchart LR
    Q["Question"] --> P["Planner<br/>sees findings so far"]
    P -->|"Next: sub-question"| W["Worker<br/>search + read"]
    W -->|"result"| P
    P -->|"Next: DONE"| S["Synthesizer"]
    S --> Ans["Predicted answer"]
    P -.->|"round 3 cap,<br/>never says DONE"| S
"""


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
