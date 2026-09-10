"""Supervisor coordinating three specialist roles — Retriever, Reasoner,
Verifier — with an explicit self-correction loop: a draft answer is only
accepted once a separate Verifier agent checks it's actually supported by
the retrieved evidence. If not, the Supervisor refines the search query
using the Verifier's stated reason and tries again.

This is a different kind of structure from orchestrator_sequential.py: that
architecture re-plans *what to look up next* to gather more facts; this one
re-plans *how to search* when what was found doesn't hold up, checked by a
role whose only job is to be skeptical of the other two.

Grounded in self-correction/self-critique loops, e.g. Reflexion
(https://arxiv.org/abs/2303.11366) and Self-Refine
(https://arxiv.org/abs/2303.17651).
"""

import json
import re
import time

from retrieval import search
from techniques.common import RunResult, llm_step, tool_step
from llm_client import call_model

NAME = "Supervisor + Verification Loop"
DESCRIPTION = (
    "A Supervisor routes between a Retriever, a Reasoner (drafts an answer from "
    "retrieved evidence), and a Verifier (checks the draft is actually supported). "
    "A rejected draft sends the Supervisor back to refine the search query, "
    "bounded to a fixed number of rounds."
)
MAX_ROUNDS = 3

REASONER_SYSTEM_PROMPT = (
    "Given the passage and the question, draft the shortest possible answer (a "
    "name, date, yes or no, etc.), and quote the exact sentence from the passage "
    "that supports it. If the passage doesn't support any answer, say so.\n"
    "Respond with exactly:\nDraft Answer: <answer>\nEvidence: <quoted sentence>"
)
VERIFIER_SYSTEM_PROMPT = (
    "You check whether a draft answer is actually supported by the evidence "
    "given for it, nothing else. If the evidence clearly supports the draft "
    "answer, respond with exactly:\nSUPPORTED\n"
    "Otherwise respond with exactly:\nNOT_SUPPORTED[<one short sentence saying what's missing>]"
)
SUPERVISOR_REFINE_PROMPT = (
    "A search did not find enough evidence to answer a question. Given the "
    "original question, the query that was tried, and what the Verifier said "
    "was missing, write one improved search query.\n"
    "Respond with exactly:\nQuery: <improved query>"
)
PROMPTS = {
    "reasoner": REASONER_SYSTEM_PROMPT,
    "verifier": VERIFIER_SYSTEM_PROMPT,
    "supervisor (query refinement)": SUPERVISOR_REFINE_PROMPT,
}

WHAT_IT_IS = (
    "A hierarchical pattern with specialist roles instead of interchangeable workers, plus a "
    "self-correction loop. The Supervisor doesn't do the retrieving, drafting, or checking "
    "itself, it routes between three roles that each do one job: a Retriever fetches evidence, "
    "a Reasoner drafts an answer from it, and a Verifier, a separate role with no stake in the "
    "draft being right, checks whether the evidence actually supports it. A rejected draft "
    "sends control back to the Supervisor to try a different search, rather than accepting the "
    "first thing the Reasoner produced. This is the reflection/self-critique idea from "
    "[Reflexion](https://arxiv.org/abs/2303.11366) and "
    "[Self-Refine](https://arxiv.org/abs/2303.17651), applied with a dedicated role for the "
    "critique step rather than asking one model to critique its own answer in the same breath "
    "it produced it."
)
HOW_WE_IMPLEMENTED_IT = (
    "A loop bounded to 3 rounds. Round 1 searches the original question; the Reasoner drafts an "
    "answer and quotes the evidence sentence it's relying on; the Verifier checks that quote "
    "against the draft and responds `SUPPORTED` or `NOT_SUPPORTED[reason]`. If supported, that "
    "draft becomes the final answer immediately. If not, a Supervisor call is shown the "
    "Verifier's stated reason and writes an improved search query for the next round. Every "
    "round costs 3 handoffs (Supervisor to Retriever, Retriever to Reasoner, Reasoner to "
    "Verifier), plus one more (Verifier back to Supervisor) if it loops again, so a run that "
    "needs all 3 rounds costs more handoffs than any other architecture here. If round 3 still "
    "isn't verified, we take that round's draft anyway and flag the run `early_terminated`."
)
WHEN_ITS_USEFUL = (
    "Reach for this when a wrong answer is more costly than a slow one, and especially when the "
    "way an answer can be wrong is 'sounds plausible but isn't actually backed by the evidence', "
    "exactly the failure mode a dedicated Verifier role is positioned to catch, since it never "
    "sees the Reasoner's confidence, only the quoted evidence. It costs more rounds and more "
    "tokens than a single pass, and our results show its real limit clearly: the refinement "
    "step only helps when a better search query exists for the Retriever to find. If the "
    "underlying tool has no better evidence no matter how the query is worded, for example a "
    "genuinely missing second hop, the loop spends its whole round budget without escaping, "
    "which is worth knowing before you lean on 'more rounds' as a fix for a retrieval problem."
)
DIAGRAM = """flowchart LR
    Q["Question"] --> Sup["Supervisor"]
    Sup --> R["Retriever<br/>search()"]
    R --> Rs["Reasoner<br/>drafts answer + evidence"]
    Rs --> V["Verifier"]
    V -->|"SUPPORTED"| Ans["Predicted answer"]
    V -->|"NOT_SUPPORTED[reason]"| Sup2["Supervisor<br/>refines query"]
    Sup2 --> R
    V -.->|"round 3 cap,<br/>still not supported"| Ans
"""

DRAFT_RE = re.compile(r"Draft Answer:\s*(.+)", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"Evidence:\s*(.+)", re.IGNORECASE | re.DOTALL)
NOT_SUPPORTED_RE = re.compile(r"NOT_SUPPORTED\[(.+?)\]", re.IGNORECASE | re.DOTALL)
QUERY_RE = re.compile(r"Query:\s*(.+)", re.IGNORECASE)


def _draft_answer(question: str, passage_title: str, passage_text: str):
    messages = [
        {"role": "system", "content": REASONER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage ({passage_title}): {passage_text}\n\nQuestion: {question}"},
    ]
    call = call_model(messages)
    content = call.content or ""
    draft_match = DRAFT_RE.search(content)
    evidence_match = EVIDENCE_RE.search(content)
    draft = draft_match.group(1).strip() if draft_match else (content.strip() or None)
    evidence = evidence_match.group(1).strip() if evidence_match else ""
    return call, draft, evidence


def _verify(question: str, draft: str | None, evidence: str):
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nDraft answer: {draft}\nEvidence: {evidence}"},
    ]
    call = call_model(messages)
    content = (call.content or "").upper()
    if "NOT_SUPPORTED" in content:
        reason_match = NOT_SUPPORTED_RE.search(call.content or "")
        return call, False, reason_match.group(1).strip() if reason_match else "unspecified"
    if "SUPPORTED" in content:
        return call, True, None
    return call, False, "verifier gave an unparseable verdict"


def _refine_query(question: str, previous_query: str, reason: str):
    messages = [
        {"role": "system", "content": SUPERVISOR_REFINE_PROMPT},
        {"role": "user", "content": f"Original question: {question}\nPrevious query: {previous_query}\nMissing: {reason}"},
    ]
    call = call_model(messages)
    match = QUERY_RE.search(call.content or "")
    new_query = match.group(1).strip() if match else previous_query
    return call, new_query


def run(example: dict, corpus) -> RunResult:
    question = example["question"]
    steps = []
    round_history = []
    query = question
    predicted_answer = None
    handoffs = 0
    used_up_budget = True
    started_at = time.perf_counter()

    for round_index in range(1, MAX_ROUNDS + 1):
        handoffs += 1  # supervisor -> retriever
        tool_started_at = time.perf_counter()
        try:
            result = search(corpus, query)
            tool_error = None
        except Exception as exc:  # noqa: BLE001 - a retrieval failure ends this round, not the run
            result, tool_error = None, str(exc)
        tool_latency = time.perf_counter() - tool_started_at
        steps.append(tool_step("supervisor", query, result, tool_latency, tool_error))

        if result is None:
            draft, evidence = None, ""
        else:
            handoffs += 1  # retriever -> reasoner
            reasoner_call, draft, evidence = _draft_answer(question, result["title"], result["text"])
            steps.append(llm_step("reasoner", f"round {round_index}", reasoner_call))

        handoffs += 1  # reasoner -> verifier
        verifier_call, supported, reason = _verify(question, draft, evidence)
        steps.append(llm_step("verifier", f"round {round_index}", verifier_call))
        round_history.append({"query": query, "draft": draft, "supported": supported})

        if supported:
            predicted_answer = draft
            used_up_budget = False
            break
        if round_index == MAX_ROUNDS:
            predicted_answer = draft
            break

        handoffs += 1  # verifier -> supervisor
        refine_call, query = _refine_query(question, query, reason)
        steps.append(llm_step("supervisor", f"refine after round {round_index}", refine_call))

    return RunResult(
        predicted_answer=predicted_answer,
        steps=steps,
        handoffs=handoffs,
        early_terminated=used_up_budget or predicted_answer is None,
        state_overhead_bytes=len(json.dumps({"question": question, "rounds": round_history}).encode("utf-8")),
        wall_clock_seconds=time.perf_counter() - started_at,
    )
