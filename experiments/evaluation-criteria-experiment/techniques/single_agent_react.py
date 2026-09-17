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
    "One agent loops Thought, Action, Observation, deciding at every turn "
    "whether to search again or finish. No decomposition, no other agents. "
    "This is the baseline every other architecture is measured against."
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

WHAT_IT_IS = (
    "ReAct (Reasoning + Acting) puts reasoning and tool use in one running transcript, produced "
    "by one model, one call at a time. Each turn the model writes a short 'Thought' (why it's "
    "doing what it's about to do) and then an 'Action' (a tool call, or an answer). Whatever the "
    "tool returns gets appended as an 'Observation', and the whole thing feeds back into the next "
    "turn's prompt. There's no separate planner, no separate worker: the same model that reasons "
    "about the problem is the same model that decides which tool to call and when to stop. "
    "The name comes from [Yao et al., 2022](https://arxiv.org/abs/2210.03629), who showed that "
    "interleaving reasoning traces with actions, rather than reasoning first and acting after "
    "or acting without reasoning at all, reduced hallucinated intermediate steps and made it "
    "easier to recover from a bad tool result, because the model's own reasoning about a "
    "surprising observation is visible in the transcript."
)
HOW_WE_IMPLEMENTED_IT = (
    "One system prompt (shown below) describes the `search[query]` and `finish[answer]` actions "
    "and the exact line format expected. `run()` keeps a single `messages` list: the system "
    "prompt, the question, and then one assistant message and one observation message appended "
    "per turn. Each turn we call the model once, regex-parse its `Action:` line, and branch: "
    "`finish[...]` sets the predicted answer and ends the run; `search[...]` calls the shared "
    "`search()` tool and appends its result as an `Observation:` message before looping again. "
    "There's a hard cap of 4 turns (`MAX_TURNS`): if the model never emits `finish[...]` by then, "
    "the run ends with no answer, flagged `early_terminated`. Because there's only one role, "
    "`handoffs` is 0 by construction, and the run's state overhead is the entire `messages` list "
    "serialized, not a small structured summary the way the multi-agent architectures pass "
    "between roles."
)
WHEN_ITS_USEFUL = (
    "Reach for this when you don't know the shape of the task ahead of time and want the model "
    "to decide, turn by turn, how much work it needs to do; when you have a small number of "
    "tools and don't want the engineering cost of coordinating separate roles; or when you're "
    "prototyping and want the fastest thing to build, since it's one prompt and one loop, easiest "
    "to debug because there's only one transcript to read top to bottom. It's a weaker fit once "
    "the task has a genuinely fixed shape (then a pipeline is more predictable and cheaper), or "
    "once correctness matters enough to want a second, independent role checking the first role's "
    "work (then look at the Supervisor pattern): a single agent never gets a second opinion on "
    "its own reasoning, and as our results show, deciding when to stop is exactly where it "
    "struggled."
)
DIAGRAM = """flowchart LR
    Q["Question"] --> A["Agent<br/>Thought + Action"]
    A -->|"search[query]"| T["search() tool"]
    T -->|"Observation"| A
    A -->|"finish[answer]"| Ans["Predicted answer"]
    A -.->|"turn 4 cap, no finish"| None["No answer<br/>(early terminated)"]
"""

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
