"""Building blocks shared by every technique: the output-format instruction
and the label catalog text that gets embedded in each prompt."""

import json

from label_catalog import INTENT_DEFINITIONS
from taxonomy import COARSE_GROUPS

RESPONSE_FORMAT_INSTRUCTION = (
    "Respond with only a JSON object of the exact form "
    '{"coarse_group": "<one of the group names above, verbatim>", '
    '"intent": "<one of that group\'s intent labels above, verbatim>"}. '
    "No other text before or after the JSON."
)


def render_label_catalog(groups: dict[str, list[str]] | None = None, with_definitions: bool = False) -> str:
    groups = groups if groups is not None else COARSE_GROUPS
    lines = []
    for group, intents in groups.items():
        lines.append(f"- {group}")
        for intent in intents:
            if with_definitions:
                lines.append(f"    - {intent}: {INTENT_DEFINITIONS[intent]}")
            else:
                lines.append(f"    - {intent}")
    return "\n".join(lines)


def render_group_names() -> str:
    return "\n".join(f"- {group}" for group in COARSE_GROUPS)


def parse_json_response(raw_text: str | None) -> dict | None:
    if not raw_text:
        return None
    text = raw_text.strip().strip("`")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def format_examples(examples: list[dict]) -> str:
    return "\n".join(
        f'Query: "{example["text"]}"\n'
        f'{{"coarse_group": "{example["coarse_group"]}", "intent": "{example["intent"]}"}}'
        for example in examples
    )
