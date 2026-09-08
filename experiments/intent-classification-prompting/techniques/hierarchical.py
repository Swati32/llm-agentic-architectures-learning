"""Decomposes the decision into two steps: first pick the coarse group,
then pick the fine intent from only that group's candidates. Unlike the
other techniques, the coarse decision here actually constrains the fine
one, rather than being reported alongside an independently-made choice."""

from llm_client import call_model
from prompt_common import parse_json_response, render_group_names, render_label_catalog
from results_schema import TechniqueRun
from taxonomy import COARSE_GROUPS, resolve_group, resolve_intent

NAME = "Hierarchical (coarse-to-fine)"
DESCRIPTION = (
    "Two model calls: first choose the coarse group, then choose the fine "
    "intent from only that group's candidates. The one technique where the "
    "coarse decision actually narrows the fine one, rather than being "
    "reported independently alongside it."
)

GROUP_STAGE_PROMPT = (
    "You are an intent classifier for a bank's customer support queries. "
    "First, decide which coarse group the query belongs to:\n\n"
    f"{render_group_names()}\n\n"
    'Respond with only {"coarse_group": "<one of the group names above, verbatim>"}.'
)


def _intent_stage_prompt(group: str) -> str:
    return (
        f"The customer's query belongs to the '{group}' group. Now choose "
        "exactly one specific intent from this list:\n\n"
        f"{render_label_catalog(groups={group: COARSE_GROUPS[group]})}\n\n"
        'Respond with only {"intent": "<one of the intent labels above, verbatim>"}.'
    )


def run(query_text: str, resources) -> TechniqueRun:
    group_call = call_model(
        [
            {"role": "system", "content": GROUP_STAGE_PROMPT},
            {"role": "user", "content": f'Query: "{query_text}"'},
        ]
    )
    group_parsed = parse_json_response(group_call.content)
    predicted_group = resolve_group(group_parsed["coarse_group"]) if group_parsed and "coarse_group" in group_parsed else None

    if predicted_group is None:
        return TechniqueRun(
            predicted_intent=None,
            predicted_group_stated=None,
            raw_response=group_call.content,
            calls=[group_call],
        )

    intent_call = call_model(
        [
            {"role": "system", "content": _intent_stage_prompt(predicted_group)},
            {"role": "user", "content": f'Query: "{query_text}"'},
        ]
    )
    intent_parsed = parse_json_response(intent_call.content)
    predicted_intent = resolve_intent(intent_parsed["intent"]) if intent_parsed and "intent" in intent_parsed else None

    return TechniqueRun(
        predicted_intent=predicted_intent,
        predicted_group_stated=predicted_group,
        raw_response=f"{group_call.content}\n---\n{intent_call.content}",
        calls=[group_call, intent_call],
    )
