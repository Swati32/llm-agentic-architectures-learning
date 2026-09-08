"""Zero-shot, but every intent in the catalog carries a one-line definition.
Tests whether disambiguating semantically close labels (e.g. card_not_working
vs virtual_card_not_working) helps without spending tokens on examples."""

from llm_client import call_model
from prompt_common import RESPONSE_FORMAT_INSTRUCTION, render_label_catalog
from results_schema import TechniqueRun, resolve_from_model_output

NAME = "Zero-shot, schema-guided"
DESCRIPTION = (
    "Same as zero-shot, but each intent is annotated with a one-line "
    "definition. Tests whether disambiguating close classes helps without "
    "the token cost of examples."
)

SYSTEM_PROMPT = (
    "You are an intent classifier for a bank's customer support queries. "
    "Classify the customer's query into exactly one coarse group and exactly "
    "one intent within that group, from this catalog:\n\n"
    f"{render_label_catalog(with_definitions=True)}\n\n"
    f"{RESPONSE_FORMAT_INSTRUCTION}"
)


def run(query_text: str, resources) -> TechniqueRun:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Query: "{query_text}"'},
    ]
    call = call_model(messages)
    return resolve_from_model_output(call.content, [call])
