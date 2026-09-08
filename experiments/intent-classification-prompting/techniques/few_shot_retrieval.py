"""Few-shot examples are chosen per query, by embedding similarity to the
training pool, instead of a single fixed set. A semantic cache reuses the
prediction for a near-duplicate query without calling the model again."""

from llm_client import call_model
from prompt_common import (
    RESPONSE_FORMAT_INSTRUCTION,
    format_examples,
    render_label_catalog,
)
from results_schema import TechniqueRun, resolve_from_model_output

RETRIEVED_EXAMPLE_COUNT = 5

NAME = "Few-shot, semantic retrieval"
DESCRIPTION = (
    "Retrieves the 5 nearest training examples to the query by embedding "
    "similarity and uses them as few-shot demonstrations, so every query "
    "gets examples relevant to it rather than a fixed, unrelated set. A "
    "semantic cache also reuses the prediction for a near-duplicate query "
    "instead of calling the model again."
)


def _build_system_prompt(retrieved_examples: list[dict]) -> str:
    examples_section = (
        f"Here are some example classifications for similar queries:\n{format_examples(retrieved_examples)}\n\n"
        if retrieved_examples
        else ""
    )
    return (
        "You are an intent classifier for a bank's customer support queries. "
        "Classify the customer's query into exactly one coarse group and "
        "exactly one intent within that group, from this catalog:\n\n"
        f"{render_label_catalog()}\n\n"
        f"{examples_section}"
        f"{RESPONSE_FORMAT_INSTRUCTION}"
    )


def run(query_text: str, resources) -> TechniqueRun:
    cached_prediction = resources.semantic_cache.lookup(query_text)
    if cached_prediction is not None:
        return TechniqueRun(
            predicted_intent=cached_prediction["predicted_intent"],
            predicted_group_stated=cached_prediction["predicted_group_stated"],
            raw_response=cached_prediction["raw_response"],
            calls=[],
            extra={"cache_hit": True, "retrieval_empty": False},
        )

    retrieval = resources.retriever.retrieve(query_text, k=RETRIEVED_EXAMPLE_COUNT)
    messages = [
        {"role": "system", "content": _build_system_prompt(retrieval.examples)},
        {"role": "user", "content": f'Query: "{query_text}"'},
    ]
    call = call_model(messages)
    result = resolve_from_model_output(
        call.content,
        [call],
        extra={"cache_hit": False, "retrieval_empty": retrieval.is_empty},
    )

    resources.semantic_cache.store(
        query_text,
        {
            "predicted_intent": result.predicted_intent,
            "predicted_group_stated": result.predicted_group_stated,
            "raw_response": result.raw_response,
        },
    )
    return result
