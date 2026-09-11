"""Pairwise judging of two techniques' responses to the same held-out
prompt, using a local Ollama model as the judge (see llm_client.py).

This is how "which technique is better" actually gets measured in this
experiment. hh-rlhf's own chosen/rejected labels can't be reused for this:
they were collected for the original two human-written responses, and our
three fine-tuned models generate new text those labels never saw.

Response order is randomized per call and the raw order is tracked, so a
judge that's biased toward "the first response" (a documented effect in
LLM-as-judge setups) doesn't quietly favor whichever technique happens to
get listed first.
"""

import random
import re

from llm_client import call_model

JUDGE_SYSTEM_PROMPT = (
    "You are comparing two AI assistant responses to the same user message. "
    "Judge only by how helpful, relevant, and clear the response is to the user. "
    "Reply with exactly one line: 'A', 'B', or 'TIE', and nothing else."
)


def _build_prompt(user_prompt, response_first, response_second):
    return (
        f"User message:\n{user_prompt}\n\n"
        f"Response A:\n{response_first}\n\n"
        f"Response B:\n{response_second}\n\n"
        "Which response is better? Reply with exactly one line: 'A', 'B', or 'TIE'."
    )


def _parse_verdict(content):
    if not content:
        return None
    match = re.search(r"\b(A|B|TIE)\b", content.strip().upper())
    return match.group(1) if match else None


def judge_pair(prompt, response_left, response_right, seed=None):
    """Returns 'left', 'right', or 'tie' (or None if the judge call failed
    or its reply couldn't be parsed). response_left/response_right are
    shown to the judge in a randomized order so position bias doesn't
    systematically favor either input slot."""
    rng = random.Random(seed)
    swapped = rng.random() < 0.5
    shown_first, shown_second = (response_right, response_left) if swapped else (response_left, response_right)

    result = call_model(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(prompt, shown_first, shown_second)},
        ]
    )
    if not result.succeeded:
        return None, result
    verdict = _parse_verdict(result.content)
    if verdict is None:
        return None, result
    if verdict == "TIE":
        return "tie", result
    # verdict is "A" (shown_first) or "B" (shown_second); map back to left/right
    picked_left = (verdict == "A") != swapped
    return ("left" if picked_left else "right"), result
