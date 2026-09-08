"""Shared, built-once objects that some techniques need (embeddings index,
semantic cache, a static example set) and others simply ignore."""

import random
from dataclasses import dataclass

from retrieval import ExampleRetriever, SemanticCache

FIXED_FEW_SHOT_SEED = 0


@dataclass
class Resources:
    train_pool: list[dict]
    retriever: ExampleRetriever
    semantic_cache: SemanticCache
    fixed_few_shot_examples: list[dict]


def _select_one_example_per_group(train_pool: list[dict]) -> list[dict]:
    examples_by_group: dict[str, list[dict]] = {}
    for example in train_pool:
        examples_by_group.setdefault(example["coarse_group"], []).append(example)
    rng = random.Random(FIXED_FEW_SHOT_SEED)
    return [rng.choice(examples) for examples in examples_by_group.values()]


def build_resources(train_pool: list[dict]) -> Resources:
    return Resources(
        train_pool=train_pool,
        retriever=ExampleRetriever(train_pool),
        semantic_cache=SemanticCache(),
        fixed_few_shot_examples=_select_one_example_per_group(train_pool),
    )
