"""Shared plumbing every architecture builds on: the step/run record shape,
and the LLM sub-tasks (reader, decomposer, synthesizer) that more than one
architecture reuses verbatim. Keeping these identical across architectures
means any difference in the results comes from *control flow* — who decides
what to search for next, and when to stop — not from different wording of
the underlying sub-tasks.
"""

import re
from dataclasses import dataclass, field

from llm_client import CallResult, call_model


@dataclass
class Step:
    role: str  # e.g. "agent", "planner", "worker", "reasoner", "verifier", "synthesizer", "retriever"
    kind: str  # "llm_call" | "tool_call"
    detail: str  # short label: the query, sub-question, or decision made
    output: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    time_to_first_token_seconds: float | None = None
    context_payload_bytes: int = 0
    is_gold_retrieval: bool | None = None  # only set for tool_call steps
    error: str | None = None


def llm_step(role: str, detail: str, call: CallResult) -> Step:
    return Step(
        role=role,
        kind="llm_call",
        detail=detail,
        output=call.content,
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
        latency_seconds=call.latency_seconds,
        time_to_first_token_seconds=call.time_to_first_token_seconds,
        context_payload_bytes=call.context_payload_bytes,
        error=call.error,
    )


def tool_step(role: str, query: str, result: dict | None, latency_seconds: float, error: str | None) -> Step:
    return Step(
        role=role,
        kind="tool_call",
        detail=query,
        output=result["text"] if result else None,
        latency_seconds=latency_seconds,
        is_gold_retrieval=result["is_gold"] if result else None,
        error=error,
    )


@dataclass
class RunResult:
    predicted_answer: str | None
    steps: list[Step]
    handoffs: int  # agent-to-agent (or role-to-role) control transfers
    early_terminated: bool  # hit the step cap without a clean finish/verdict
    state_overhead_bytes: int  # size of the shared state serialized at the final handoff
    wall_clock_seconds: float  # actual elapsed time (differs from total_latency_seconds when parallel)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind == "tool_call")

    @property
    def tool_errors(self) -> int:
        return sum(1 for step in self.steps if step.kind == "tool_call" and step.error)

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
        first_llm_step = next((step for step in self.steps if step.kind == "llm_call"), None)
        return first_llm_step.time_to_first_token_seconds if first_llm_step else None

    @property
    def mean_context_payload_bytes(self) -> float:
        llm_steps = [step for step in self.steps if step.kind == "llm_call"]
        return sum(step.context_payload_bytes for step in llm_steps) / len(llm_steps) if llm_steps else 0.0

    @property
    def empty_retrievals(self) -> int:
        """"Empty" here means the retrieved paragraph wasn't one of the two
        gold paragraphs for the question — search1 with k=1 never returns
        literally nothing, so an irrelevant top-1 hit is this task's
        equivalent of an empty/wasted retrieval."""
        return sum(1 for step in self.steps if step.kind == "tool_call" and step.is_gold_retrieval is False)

    @property
    def empty_retrieval_rate(self) -> float:
        return self.empty_retrievals / self.tool_calls if self.tool_calls else 0.0

    @property
    def had_error(self) -> bool:
        return any(step.error for step in self.steps)


# ---- Shared sub-tasks -------------------------------------------------

DECOMPOSER_SYSTEM_PROMPT = (
    "Break the following multi-hop question into exactly 2 ordered sub-questions "
    "such that answering them in order gives enough information to answer the "
    "original question. Each sub-question must be self-contained: do not use a "
    "pronoun like 'it', 'he', or 'that team' to refer to something not yet identified.\n"
    "Respond with exactly:\n1. <sub-question 1>\n2. <sub-question 2>"
)
SUBQUESTION_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)


def decompose(question: str) -> tuple[CallResult, list[str]]:
    messages = [
        {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    call = call_model(messages)
    sub_questions = SUBQUESTION_LINE_RE.findall(call.content or "")
    if len(sub_questions) < 2:
        # Degrade gracefully rather than crash the run: ask the full
        # question in both hops so the architecture can still finish.
        sub_questions = [question, question]
    return call, [q.strip() for q in sub_questions[:2]]


READER_SYSTEM_PROMPT = (
    "Answer the question using only the passage given. Give the shortest correct "
    "answer (a name, date, yes or no, etc.), not a full sentence. If the passage "
    "does not contain the answer, say so.\n"
    "Respond with exactly:\nAnswer: <answer>"
)
ANSWER_LINE_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)


def read_answer(sub_question: str, passage_title: str, passage_text: str) -> tuple[CallResult, str | None]:
    messages = [
        {"role": "system", "content": READER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage ({passage_title}): {passage_text}\n\nQuestion: {sub_question}"},
    ]
    call = call_model(messages)
    match = ANSWER_LINE_RE.search(call.content or "")
    answer = match.group(1).strip() if match else (call.content.strip() if call.content else None)
    return call, answer


SYNTHESIZER_SYSTEM_PROMPT = (
    "You combine the results of a multi-step research process into one final "
    "answer. Given the original question and a list of sub-questions with their "
    "answers, give the shortest correct final answer (a name, date, yes or no, "
    "etc.), not a full sentence.\n"
    "Respond with exactly:\nFinal Answer: <answer>"
)
FINAL_ANSWER_LINE_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE)


def synthesize(question: str, sub_qas: list[tuple[str, str | None]]) -> tuple[CallResult, str | None]:
    findings = "\n".join(f"- {sub_q} -> {sub_a}" for sub_q, sub_a in sub_qas)
    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Original question: {question}\n\nFindings:\n{findings}"},
    ]
    call = call_model(messages)
    match = FINAL_ANSWER_LINE_RE.search(call.content or "")
    answer = match.group(1).strip() if match else (call.content.strip() if call.content else None)
    return call, answer
