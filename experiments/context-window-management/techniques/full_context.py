"""Baseline: put the entire document into the prompt, unmodified, no
matter how long it is. No decision is made about what to keep, everything
goes in. This is the floor every other technique here is measured
against: does deciding what to keep in context actually buy anything over
never having to decide at all?
"""

import time

from techniques.common import ANSWER_SYSTEM_PROMPT, RunResult, generate_answer, llm_step

NAME = "Full Context"
DESCRIPTION = "Every word of the document goes into the prompt, unmodified. No management at all."
PROMPTS = {"generator": ANSWER_SYSTEM_PROMPT}

WHAT_IT_IS = (
    "The strategy every other technique in this experiment is measured against: whatever the "
    "document is, all of it goes into the prompt, in its original order, with no cuts, no "
    "compression, and no relevance judgment. If the context window is big enough, this always "
    "works from a 'did the information reach the model' standpoint. Whether the model actually "
    "*uses* information sitting in the middle of a long prompt as well as it uses information "
    "near the start or end is a separate question. [Liu et al., 2023](https://arxiv.org/abs/2307.03172) "
    "found it often doesn't, a pattern known as 'lost in the middle'."
)
HOW_WE_IMPLEMENTED_IT = (
    "The document text is passed straight into the shared `generate_answer()` prompt (see "
    "[techniques/common.py](common.py)) with no processing at all. One LLM call, regardless of "
    "document length."
)
WHEN_ITS_USEFUL = (
    "Reach for this whenever the content actually fits comfortably inside the model's context "
    "window and cost isn't a concern: it's the simplest strategy, has no extra moving parts to "
    "get wrong, and loses no information by construction. It stops being the right choice once "
    "the content genuinely doesn't fit, the model can't be given tokens that don't exist, and "
    "even before that hard limit, once the model's *effective* use of a long prompt starts "
    "degrading well short of running out of room."
)
DIAGRAM = """flowchart LR
    D["Document (any length)"] --> G["Generate answer<br/>from full document"]
    G --> Ans["Predicted answer"]
"""


def run(document: str, question: str) -> RunResult:
    started_at = time.perf_counter()
    call, answer = generate_answer(question, document)
    steps = [llm_step("generator", "generate answer from full document", call)]
    return RunResult(
        predicted_answer=answer,
        steps=steps,
        context_sent_to_generator=document,
        wall_clock_seconds=time.perf_counter() - started_at,
    )
