"""A small, static set of examples (one per coarse group, chosen once) is
included in every prompt regardless of the query. With 77 intents, this
set can't cover every class — it's the naive-few-shot baseline that
technique 4 (retrieval-based few-shot) is meant to improve on."""

from llm_client import call_model
from prompt_common import (
    RESPONSE_FORMAT_INSTRUCTION,
    format_examples,
    render_label_catalog,
)
from results_schema import TechniqueRun, resolve_from_model_output

NAME = "Few-shot, fixed examples"
DESCRIPTION = (
    "A static set of examples (one per coarse group) is included in every "
    "prompt. Can't cover all 77 intents with examples, so it mostly "
    "demonstrates the expected output format and a handful of intents."
)


def _build_system_prompt(fixed_examples: list[dict]) -> str:
    return (
        "You are an intent classifier for a bank's customer support queries. "
        "Classify the customer's query into exactly one coarse group and "
        "exactly one intent within that group, from this catalog:\n\n"
        f"{render_label_catalog()}\n\n"
        "Here are some example classifications:\n"
        f"{format_examples(fixed_examples)}\n\n"
        f"{RESPONSE_FORMAT_INSTRUCTION}"
    )


def run(query_text: str, resources) -> TechniqueRun:
    messages = [
        {"role": "system", "content": _build_system_prompt(resources.fixed_few_shot_examples)},
        {"role": "user", "content": f'Query: "{query_text}"'},
    ]
    call = call_model(messages)
    return resolve_from_model_output(call.content, [call])
