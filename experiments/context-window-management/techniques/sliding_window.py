"""Keeps only the most recent WINDOW_TOKENS words of the document, drops
everything before that, and answers from what remains. No judgment about
relevance is made, only recency. This is the cheapest way to bound
context size for anything that grows over time (a long chat history, a
stream of tool output), but it makes a specific bet: whatever mattered is
recent.
"""

import time

from techniques.common import ANSWER_SYSTEM_PROMPT, RunResult, generate_answer, llm_step

NAME = "Sliding Window"
DESCRIPTION = "Keeps only the last WINDOW_TOKENS words of the document; everything older is dropped."
WINDOW_TOKENS = 1500
PROMPTS = {"generator": ANSWER_SYSTEM_PROMPT}

WHAT_IT_IS = (
    f"Truncates the document down to its last {WINDOW_TOKENS} words before generating an answer, "
    "the same principle a chat app uses when it only resends the last N turns of a conversation, "
    "or a log viewer that only shows the tail of a file. It makes no judgment about what's "
    "relevant, only about what's recent, on the assumption that older content matters less. "
    "Grounded in the recency-window family of context management used in streaming inference "
    "setups like [StreamingLLM](https://arxiv.org/abs/2309.17453), minus the attention-sink "
    "tokens StreamingLLM keeps specifically to stop the model's attention pattern from breaking."
)
HOW_WE_IMPLEMENTED_IT = (
    f"Splits the document on whitespace, keeps only the last {WINDOW_TOKENS} words, rejoins them, "
    "and passes that truncated text into the same `generate_answer()` prompt every technique here "
    "uses (see [techniques/common.py](common.py)). One LLM call, and the window size never adapts "
    "to where the answer actually is, that's exactly what's being tested."
)
WHEN_ITS_USEFUL = (
    "The right call when the underlying content genuinely is a stream where recent items matter "
    "most, like the tail of a live log or the last few turns of a conversation about something "
    "that just happened. It fails deterministically, not gracefully, whenever the information "
    "actually needed sits further back than the window reaches; there's no partial credit for "
    "'almost recent enough'."
)
DIAGRAM = f"""flowchart LR
    D["Document (any length)"] --> W["Keep last {WINDOW_TOKENS} words"]
    W --> G["Generate answer<br/>from window only"]
    G --> Ans["Predicted answer"]
"""


def run(document: str, question: str) -> RunResult:
    started_at = time.perf_counter()
    words = document.split()
    windowed_text = " ".join(words[-WINDOW_TOKENS:])
    call, answer = generate_answer(question, windowed_text)
    steps = [llm_step("generator", "generate answer from windowed text", call)]
    return RunResult(
        predicted_answer=answer,
        steps=steps,
        context_sent_to_generator=windowed_text,
        wall_clock_seconds=time.perf_counter() - started_at,
    )
