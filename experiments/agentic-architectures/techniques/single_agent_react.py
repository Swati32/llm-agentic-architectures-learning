"""Baseline: one agent, one loop. No decomposition, no other agents — the
model decides for itself, turn by turn, whether to search again or answer.
This is the architecture everything else is measured against: does adding
structure (fixed pipeline, orchestration, supervision) actually buy
anything over just letting one agent reason and act freely?

Grounded in ReAct (https://arxiv.org/abs/2210.03629): interleaving
"Thought" reasoning with "Action" tool calls in one running transcript.
"""

import json
import re
import time

from llm_client import call_model
from retrieval import search
from techniques.common import RunResult, llm_step, tool_step

NAME = "Single-Agent ReAct"
DESCRIPTION = (
    "One agent loops Thought -> Action -> Observation, deciding at every turn "
    "whether to search again or finish. No decomposition, no other agents — "
    "the baseline every other architecture is measured against."
)
MAX_TURNS = 4

SYSTEM_PROMPT = (
    "You are answering a question that may require looking up multiple facts. "
    "You have one tool:\n\n"
    "search[query] - returns the single most relevant passage for `query` from "
    "a fixed set of documents about this question.\n\n"
    "Work in a loop. Each turn, write exactly one Thought line and one Action line:\n"
    "Thought: <your reasoning>\n"
    "Action: search[<query>]\n\n"
    "or, once you have enough information:\n"
    "Thought: <your reasoning>\n"
    "Action: finish[<final answer>]\n\n"
    "Give the shortest correct answer (a name, date, yes or no, etc.), not a "
    "full sentence. Do not write anything after the Action line."
)
PROMPTS = {"agent": SYSTEM_PROMPT}

FINISH_ACTION_RE = re.compile(r"Action:\s*finish\[(.+?)\]", re.IGNORECASE | re.DOTALL)
SEARCH_ACTION_RE = re.compile(r"Action:\s*search\[(.+?)\]", re.IGNORECASE | re.DOTALL)


def parse_action(content: str | None) -> tuple[str, str] | None:
    if content is None:
        return None
    finish_match = FINISH_ACTION_RE.search(content)
    if finish_match:
        return "finish", finish_match.group(1).strip()
    search_match = SEARCH_ACTION_RE.search(content)
    if search_match:
        return "search", search_match.group(1).strip()
    return None


def run(example: dict, corpus) -> RunResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {example['question']}"},
    ]
    steps = []
    predicted_answer = None
    early_terminated = True
    started_at = time.perf_counter()

    for turn in range(1, MAX_TURNS + 1):
        call = call_model(messages)
        steps.append(llm_step("agent", f"turn {turn}", call))
        if call.error:
            break
        messages.append({"role": "assistant", "content": call.content})

        action = parse_action(call.content)
        if action is None:
            break
        kind, argument = action
        if kind == "finish":
            predicted_answer = argument
            early_terminated = False
            break

        tool_started_at = time.perf_counter()
        try:
            result = search(corpus, argument)
            tool_error = None
        except Exception as exc:  # noqa: BLE001 - a retrieval failure ends the turn, not the run
            result, tool_error = None, str(exc)
        tool_latency = time.perf_counter() - tool_started_at
        steps.append(tool_step("agent", argument, result, tool_latency, tool_error))
        observation = result["text"] if result else f"Error: {tool_error}"
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return RunResult(
        predicted_answer=predicted_answer,
        steps=steps,
        handoffs=0,
        early_terminated=early_terminated,
        state_overhead_bytes=len(json.dumps(messages).encode("utf-8")),
        wall_clock_seconds=time.perf_counter() - started_at,
    )
