"""Shared plumbing every technique builds on: the step/run record shape,
and the two LLM sub-tasks (final-answer generation, chunk summarization)
more than one technique uses.

The final-answer prompt (`ANSWER_SYSTEM_PROMPT`) is shared verbatim by
every technique, on purpose: any difference in how often a technique gets
the answer right comes from what it put in front of the model, not from
one technique being told to answer more carefully than another.
"""

import re
from dataclasses import dataclass

from llm_client import CallResult, call_model

ANSWER_SYSTEM_PROMPT = (
    "You are a research assistant answering a question using only the text "
    "provided below. Give the shortest correct answer (a name, date, number, "
    "etc.), not a full sentence. If the text does not contain enough "
    "information to answer the question, respond with exactly: "
    "Insufficient information.\n"
    "Respond with exactly:\nAnswer: <answer>"
)
ANSWER_LINE_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)

SUMMARIZE_SYSTEM_PROMPT = (
    "Summarize the text below in 2-4 sentences, preserving every specific "
    "fact, name, date, and number it contains. This summary will replace "
    "the original text, so anything you drop is gone for good."
)


@dataclass
class Step:
    role: str  # "summarizer" | "generator"
    detail: str
    output: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    time_to_first_token_seconds: float | None = None
    context_payload_bytes: int = 0
    error: str | None = None


def llm_step(role: str, detail: str, call: CallResult) -> Step:
    return Step(
        role=role,
        detail=detail,
        output=call.content,
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
        latency_seconds=call.latency_seconds,
        time_to_first_token_seconds=call.time_to_first_token_seconds,
        context_payload_bytes=call.context_payload_bytes,
        error=call.error,
    )


@dataclass
class RunResult:
    predicted_answer: str | None
    steps: list[Step]
    context_sent_to_generator: str  # what the final answer call actually saw, for the document explorer
    wall_clock_seconds: float

    @property
    def llm_calls(self) -> int:
        return len(self.steps)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(step.prompt_tokens for step in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        return sum(step.completion_tokens for step in self.steps)

    @property
    def total_latency_seconds(self) -> float:
        return sum(step.latency_seconds for step in self.steps)

    @property
    def time_to_first_token_seconds(self) -> float | None:
        first_step = self.steps[0] if self.steps else None
        return first_step.time_to_first_token_seconds if first_step else None

    @property
    def mean_context_payload_bytes(self) -> float:
        return sum(step.context_payload_bytes for step in self.steps) / len(self.steps) if self.steps else 0.0

    @property
    def had_error(self) -> bool:
        return any(step.error for step in self.steps)


def generate_answer(question: str, context_text: str) -> tuple[CallResult, str | None]:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Text:\n{context_text}\n\nQuestion: {question}"},
    ]
    call = call_model(messages)
    match = ANSWER_LINE_RE.search(call.content or "")
    answer = match.group(1).strip() if match else (call.content.strip() if call.content else None)
    return call, answer


def summarize_chunk(chunk_text: str, running_summary: str | None) -> tuple[CallResult, str | None]:
    """Folds `chunk_text` into `running_summary` (or starts a fresh one if
    this is the first chunk). One call per chunk boundary, mirroring how a
    real long-running agent's context gets compacted incrementally rather
    than all at once at the end."""
    if running_summary:
        user_content = (
            f"Summary so far:\n{running_summary}\n\n"
            f"New text to fold in:\n{chunk_text}\n\n"
            "Write one updated summary that covers both."
        )
    else:
        user_content = f"Text:\n{chunk_text}"
    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    call = call_model(messages)
    updated_summary = call.content.strip() if call.content else running_summary
    return call, updated_summary
