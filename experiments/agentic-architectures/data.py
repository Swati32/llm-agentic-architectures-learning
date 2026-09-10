"""HotpotQA (distractor config) loading and sampling.

Each example ships a question, a short answer, and 10 candidate paragraphs
(2 "gold" paragraphs that together support the answer, 8 unrelated
distractors). That paragraph set is used as-is as the per-question corpus a
`search()` tool retrieves over (see retrieval.py) — no live web search, so
every architecture is compared on exactly the same fixed body of evidence.

`distractor` is a Parquet-native config on the Hub (no deprecated loading
script), so it loads with current `datasets` versions unmodified.
"""

import random

from datasets import load_dataset

DATASET_REPO = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"


def load_pool(split: str = "validation") -> list[dict]:
    raw_split = load_dataset(DATASET_REPO, DATASET_CONFIG, split=split)
    pool = []
    for row in raw_split:
        titles = row["context"]["title"]
        sentence_lists = row["context"]["sentences"]
        paragraphs = [
            {"title": title, "text": " ".join(sentences).strip()}
            for title, sentences in zip(titles, sentence_lists)
        ]
        gold_titles = set(row["supporting_facts"]["title"])
        pool.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],  # "bridge" or "comparison"
                "level": row["level"],
                "paragraphs": paragraphs,
                "gold_titles": gold_titles,
            }
        )
    return pool


def stratified_sample(pool: list[dict], target_size: int, seed: int = 0) -> list[dict]:
    """Split roughly evenly across question type (bridge/comparison), since
    the two types stress decomposition differently: comparison questions
    decompose into two independent lookups, bridge questions into a lookup
    whose result is needed to form the second query."""
    by_type: dict[str, list[dict]] = {}
    for example in pool:
        by_type.setdefault(example["type"], []).append(example)

    if target_size < len(by_type):
        raise ValueError(
            f"target_size ({target_size}) is smaller than the number of "
            f"distinct question types ({len(by_type)}); can't give every type "
            "at least one slot without exceeding target_size."
        )

    rng = random.Random(seed)
    for examples in by_type.values():
        rng.shuffle(examples)

    per_type = target_size // len(by_type)
    sample = []
    for examples in by_type.values():
        sample.extend(examples[:per_type])
    rng.shuffle(sample)
    return sample
