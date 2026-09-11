"""Shared plumbing every architecture builds on: the step/run record shape,
and the LLM sub-tasks (answer generation, relevance grading, HyDE passage
generation, query decomposition) that more than one architecture uses.

The final-answer prompt (`ANSWER_SYSTEM_PROMPT`) is shared verbatim by all
six architectures, on purpose: every one of them is told, identically, to
say "Insufficient information." when the passages it was given don't
support an answer. That means any difference in how often an architecture
actually abstains on a null_query comes from what it retrieved and how
(dense vs. hybrid vs. reranked, whether it explicitly graded relevance
first), not from one architecture being told to be more careful than
another.
"""

import re
from dataclasses import dataclass, field

from llm_client import CallResult, call_model
from retrieval import RetrievedChunk


@dataclass
class Step:
    role: str  # "query_rewriter" | "retriever" | "reranker" | "grader" | "generator"
    kind: str  # "llm_call" | "retrieval_call"
    detail: str  # query text, or a short label for what the step did
    output: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    time_to_first_token_seconds: float | None = None
    context_payload_bytes: int = 0
    retrieved_chunk_ids: list[str] = field(default_factory=list)
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


def retrieval_step(role: str, query: str, chunks: list[RetrievedChunk], latency_seconds: float) -> Step:
    return Step(
        role=role,
        kind="retrieval_call",
        detail=query,
        output=", ".join(c.title for c in chunks) if chunks else None,
        latency_seconds=latency_seconds,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
    )


@dataclass
class RunResult:
    predicted_answer: str | None
    steps: list[Step]
    retrieved_chunks: list[RetrievedChunk]  # the chunks actually fed to the final answer call
    abstained: bool  # model said "Insufficient information."
    wall_clock_seconds: float
    retrieval_rounds: int = 1  # set explicitly per architecture: how many separate retrieval
    # *attempts* this run made, where a multi-query fan-out (several sub-queries searched at
    # once, e.g. query decomposition) still counts as 1 round, and a corrective re-retrieval
    # (e.g. CRAG trying again after grading its first pass insufficient) counts as a 2nd. This
    # is what separates "this architecture searches more per round" (retrieval_calls) from
    # "this architecture decided it needed to try again" (retrieval_rounds).

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def retrieval_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind == "retrieval_call")

    @property
    def llm_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind == "llm_call")

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
        """A retrieval_call step that came back with zero chunks at all
        (only possible when a sub-query search or corrective re-retrieval
        genuinely finds nothing, since top-k dense/BM25 search otherwise
        always returns k results)."""
        return sum(1 for step in self.steps if step.kind == "retrieval_call" and not step.retrieved_chunk_ids)

    @property
    def empty_retrieval_rate(self) -> float:
        return self.empty_retrievals / self.retrieval_calls if self.retrieval_calls else 0.0

    @property
    def had_error(self) -> bool:
        return any(step.error for step in self.steps)


# ---- Shared sub-tasks -------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are a research assistant answering a question using only the passages "
    "provided below. Give the shortest correct answer (a name, date, yes/no, "
    "etc.), not a full sentence. If the passages do not contain enough "
    "information to answer the question, respond with exactly: Insufficient information.\n"
    "Respond with exactly:\nAnswer: <answer>"
)
ANSWER_LINE_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


def format_passages(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i + 1}] ({chunk.title}): {chunk.text}" for i, chunk in enumerate(chunks))


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[CallResult, str | None, bool]:
    passages = format_passages(chunks) if chunks else "(no passages retrieved)"
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"},
    ]
    call = call_model(messages)
    match = ANSWER_LINE_RE.search(call.content or "")
    answer = match.group(1).strip() if match else (call.content.strip() if call.content else None)
    abstained = answer is not None and answer.strip().rstrip(".").lower() == "insufficient information"
    return call, answer, abstained


GRADER_SYSTEM_PROMPT = (
    "You grade whether each passage below is relevant enough to help answer the "
    "question. For every numbered passage, respond with exactly one line:\n"
    "<number>: relevant\nor\n<number>: irrelevant"
)
GRADE_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(relevant|irrelevant)", re.IGNORECASE | re.MULTILINE)


def grade_relevance(question: str, chunks: list[RetrievedChunk]) -> tuple[CallResult, list[bool]]:
    """Returns one bool per chunk, in the same order as `chunks`. Defaults
    a chunk to irrelevant if the model's grading line for it is missing or
    unparseable, which is the safer failure mode for a technique whose
    whole point is not over-trusting weak retrieval."""
    passages = format_passages(chunks)
    messages = [
        {"role": "system", "content": GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
    ]
    call = call_model(messages)
    verdicts_by_number = {int(n): v.lower() == "relevant" for n, v in GRADE_LINE_RE.findall(call.content or "")}
    return call, [verdicts_by_number.get(i + 1, False) for i in range(len(chunks))]


HYDE_SYSTEM_PROMPT = (
    "Write a short, plausible passage (3-4 sentences) that would answer the "
    "question below, written as if it were a snippet from a news article. Give "
    "your best guess even if you are not certain; this passage will only be "
    "used to search for real supporting evidence, and will not be shown to "
    "anyone."
)


def generate_hypothetical_passage(question: str) -> tuple[CallResult, str | None]:
    messages = [
        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    call = call_model(messages)
    return call, (call.content.strip() if call.content else None)


DECOMPOSE_SYSTEM_PROMPT = (
    "Break the following question into 2 to 3 simpler, self-contained "
    "sub-questions that together cover all the information needed to answer "
    "it. Each sub-question must stand on its own: do not use a pronoun like "
    "'it', 'he', or 'that company' to refer to something not yet identified.\n"
    "Respond with exactly:\n1. <sub-question>\n2. <sub-question>\n(3. <sub-question>, if needed)"
)
SUBQUESTION_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+)$", re.MULTILINE)


def decompose_query(question: str) -> tuple[CallResult, list[str]]:
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    call = call_model(messages)
    sub_questions = [q.strip() for q in SUBQUESTION_LINE_RE.findall(call.content or "")]
    if not sub_questions:
        # Degrade gracefully rather than crash the run: search the original
        # question if decomposition didn't parse.
        sub_questions = [question]
    return call, sub_questions[:3]
