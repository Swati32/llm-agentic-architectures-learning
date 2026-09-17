"""The plain LLM shape: one prompt, one answer, no retrieval, no tool, no
loop. The model has to answer from whatever it learned during training,
nothing else. This is the baseline every other shape in this experiment
adds one capability on top of: RAG adds a lookup, single agent adds a loop
that decides whether to look up again, multi-agent adds a second role that
checks the first role's work.

HotpotQA's supporting facts come from specific Wikipedia paragraphs, not
general knowledge, so this shape is expected to struggle: that's not a
weakness in the technique, it's the exact gap retrieval exists to close,
and this experiment's whole point is measuring how visibly each shape's
metrics show that gap, not just its final accuracy.
"""

import json
import time

from llm_client import call_model
from techniques.common import RunResult, llm_step

NAME = "Plain LLM (no retrieval)"
DESCRIPTION = (
    "One prompt, one answer, from the model's own training knowledge only. No "
    "search tool, no loop, no other role. The floor every other shape in this "
    "experiment is compared against."
)

SYSTEM_PROMPT = (
    "Answer the question using only what you already know. You have no tools "
    "and cannot look anything up. Give the shortest correct answer (a name, "
    "date, yes or no, etc.), not a full sentence. If you don't know, give your "
    "best guess rather than refusing.\n"
    "Respond with exactly:\nAnswer: <answer>"
)
PROMPTS = {"model": SYSTEM_PROMPT}

WHAT_IT_IS = (
    "A single call to the model with the question and nothing else, no passage, no tool, no "
    "second turn. Whatever the model answers comes entirely from what it memorized during "
    "training (its parametric knowledge), not from anything it looked up. This is the shape "
    "every LLM application starts as, before anyone adds retrieval or tool use to it."
)
HOW_WE_IMPLEMENTED_IT = (
    "One system prompt (shown below) telling the model it has no tools and should give its best "
    "guess rather than refuse, so a missing fact shows up as a wrong answer (comparable to every "
    "other shape's wrong answers) rather than as a different failure mode (a refusal) that the "
    "same quality metrics can't score the same way. `run()` sends one message, parses one "
    "`Answer:` line, and returns. There is no tool, so `handoffs` is 0, `early_terminated` is "
    "always `False` (there's no step cap to hit, a single call always either answers or doesn't), "
    "and retrieval-shaped metrics (empty retrieval rate, redundant step rate) are undefined "
    "(reported as 0 over 0 tool calls) rather than meaningfully 0%."
)
WHEN_ITS_USEFUL = (
    "Reach for this when the task is squarely inside general knowledge the model already has "
    "(common facts, well-known reasoning patterns, style and format tasks), where adding "
    "retrieval or a tool loop would only add latency and cost for no accuracy gain. It's the "
    "wrong shape once the task depends on information the model wasn't trained on, information "
    "that changed since training, or a private/internal corpus, which is exactly this "
    "experiment's task: this shape's accuracy here is the direct, measured cost of skipping "
    "retrieval on a task that needs it."
)
DIAGRAM = """flowchart LR
    Q["Question"] --> M["Model<br/>(training knowledge only)"]
    M --> Ans["Predicted answer"]
"""

ANSWER_PREFIX = "Answer:"


def _parse_answer(content: str | None) -> str | None:
    if content is None:
        return None
    for line in content.splitlines():
        line = line.strip()
        if line.lower().startswith(ANSWER_PREFIX.lower()):
            return line[len(ANSWER_PREFIX):].strip()
    return content.strip() or None


def run(example: dict, corpus) -> RunResult:  # noqa: ARG001 - corpus unused, kept for a uniform run() signature across shapes
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {example['question']}"},
    ]
    started_at = time.perf_counter()
    call = call_model(messages)
    steps = [llm_step("model", "answer from training knowledge", call)]
    predicted_answer = None if call.error else _parse_answer(call.content)

    return RunResult(
        predicted_answer=predicted_answer,
        steps=steps,
        handoffs=0,
        early_terminated=False,
        state_overhead_bytes=len(json.dumps(messages).encode("utf-8")),
        wall_clock_seconds=time.perf_counter() - started_at,
    )
