"""Plain instruction + the full label catalog, no examples. The cheapest,
fastest technique — the baseline everything else is measured against."""

from llm_client import call_model
from prompt_common import RESPONSE_FORMAT_INSTRUCTION, render_label_catalog
from results_schema import TechniqueRun, resolve_from_model_output

NAME = "Zero-shot"
DESCRIPTION = (
    "Instruction plus the list of coarse groups and intents, with no examples "
    "and no per-label definitions. The baseline: cheapest and fastest, relies "
    "entirely on the model's prior knowledge of what each label name means."
)

SYSTEM_PROMPT = (
    "You are an intent classifier for a bank's customer support queries. "
    "Classify the customer's query into exactly one coarse group and exactly "
    "one intent within that group, from this catalog:\n\n"
    f"{render_label_catalog()}\n\n"
    f"{RESPONSE_FORMAT_INSTRUCTION}"
)


def run(query_text: str, resources) -> TechniqueRun:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Query: "{query_text}"'},
    ]
    call = call_model(messages)
    return resolve_from_model_output(call.content, [call])
