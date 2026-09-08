"""Samples chain-of-thought reasoning multiple times at a non-zero
temperature and takes the majority answer. The expensive end of the
spectrum: N model calls instead of 1, in exchange for outvoting a
one-off reasoning mistake."""

from collections import Counter

from llm_client import call_model
from results_schema import TechniqueRun, resolve_from_model_output
from techniques.chain_of_thought import SYSTEM_PROMPT

SAMPLE_COUNT = 3
SAMPLING_TEMPERATURE = 0.7

NAME = "Self-consistency"
DESCRIPTION = (
    f"Samples chain-of-thought {SAMPLE_COUNT} times at temperature "
    f"{SAMPLING_TEMPERATURE} and takes the majority vote on both fields. "
    "Costs N calls instead of 1 — the question is whether the accuracy "
    "gain over plain chain-of-thought is worth that multiple."
)


def _majority(values: list[str | None]) -> str | None:
    votes = Counter(value for value in values if value is not None)
    return votes.most_common(1)[0][0] if votes else None


def run(query_text: str, resources) -> TechniqueRun:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Query: "{query_text}"'},
    ]
    calls = [call_model(messages, temperature=SAMPLING_TEMPERATURE) for _ in range(SAMPLE_COUNT)]
    samples = [resolve_from_model_output(call.content, [call]) for call in calls]

    return TechniqueRun(
        predicted_intent=_majority([sample.predicted_intent for sample in samples]),
        predicted_group_stated=_majority([sample.predicted_group_stated for sample in samples]),
        raw_response="\n---\n".join(sample.raw_response or "" for sample in samples),
        calls=calls,
        extra={"sample_intents": [sample.predicted_intent for sample in samples]},
    )
