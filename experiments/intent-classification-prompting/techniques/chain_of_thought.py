"""Asks the model to reason before answering. Intent classification is a
single-step decision, so the interesting question is whether reasoning
actually improves accuracy here or just adds latency and tokens."""

from llm_client import call_model
from prompt_common import RESPONSE_FORMAT_INSTRUCTION, render_label_catalog
from results_schema import TechniqueRun, resolve_from_model_output

NAME = "Chain-of-thought"
DESCRIPTION = (
    "Asks the model to reason step by step before answering. Classification "
    "is a single-step decision with no intermediate state, so this tests "
    "whether reasoning helps at all here, or only adds latency and tokens."
)

SYSTEM_PROMPT = (
    "You are an intent classifier for a bank's customer support queries. "
    "Classify the customer's query into exactly one coarse group and exactly "
    "one intent within that group, from this catalog:\n\n"
    f"{render_label_catalog()}\n\n"
    "First, reason step by step about what the customer is asking, in a "
    "few sentences. Then, on the final line, give your answer as JSON.\n"
    f"{RESPONSE_FORMAT_INSTRUCTION}"
)


def run(query_text: str, resources) -> TechniqueRun:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Query: "{query_text}"'},
    ]
    call = call_model(messages)
    return resolve_from_model_output(call.content, [call])
