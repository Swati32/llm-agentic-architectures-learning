"""Banking77 loading and sampling.

Loaded from `legacy-datasets/banking77` rather than the canonical
`PolyAI/banking77`: the canonical repo still ships a loading script, which
current versions of the `datasets` library refuse to execute. The legacy
mirror holds the same data, pre-converted to Parquet.
"""

import random

from datasets import load_dataset

from taxonomy import INTENT_TO_GROUP

DATASET_REPO = "legacy-datasets/banking77"


def load_intent_labels() -> list[str]:
    train_split = load_dataset(DATASET_REPO, split="train")
    return train_split.features["label"].names


def load_examples(split: str) -> list[dict]:
    """Each example: {"text": ..., "intent": ..., "coarse_group": ...}."""
    intent_labels = load_intent_labels()
    raw_split = load_dataset(DATASET_REPO, split=split)
    examples = []
    for row in raw_split:
        intent = intent_labels[row["label"]]
        examples.append(
            {
                "text": row["text"],
                "intent": intent,
                "coarse_group": INTENT_TO_GROUP[intent],
            }
        )
    return examples


def stratified_sample(examples: list[dict], target_size: int, seed: int = 0) -> list[dict]:
    """Draw a sample proportional to each intent's share of `examples`,
    so every technique is judged on a set that reflects the full label
    distribution rather than an accident of random sampling.

    Requires target_size >= the number of distinct intents present, since
    every intent gets at least one slot; a smaller target_size can't be
    stratified across all of them and raises instead of silently growing
    the sample past what was asked for."""
    by_intent: dict[str, list[dict]] = {}
    for example in examples:
        by_intent.setdefault(example["intent"], []).append(example)

    if target_size < len(by_intent):
        raise ValueError(
            f"target_size ({target_size}) is smaller than the number of "
            f"distinct intents ({len(by_intent)}); can't give every intent "
            "at least one slot without exceeding target_size."
        )

    rng = random.Random(seed)
    sample = []
    for intent_examples in by_intent.values():
        share = max(1, round(target_size * len(intent_examples) / len(examples)))
        sample.extend(rng.sample(intent_examples, min(share, len(intent_examples))))

    rng.shuffle(sample)
    return sample
