"""SQuAD loading and synthetic needle-in-haystack document construction.

SQuAD (https://rajpurkar.github.io/SQuAD-explorer/) ships short Wikipedia
paragraphs, each paired with a question and a short extractive answer.
This experiment doesn't use SQuAD's paragraphs as-is, they're too short
to stress a context window on their own. Instead it treats one paragraph
as a "needle" (the one piece of text that actually contains the answer)
and buries it inside a pile of other, unrelated SQuAD paragraphs (the
"haystack") at a controlled position and controlled total length. That's
the same synthetic construction the "Lost in the Middle" paper
(https://arxiv.org/abs/2307.03172) and most needle-in-a-haystack context
evals use: real short-form QA, artificially long surrounding context.
"""

import random

from datasets import load_dataset

DATASET_REPO = "rajpurkar/squad"


def load_examples(split: str = "train") -> list[dict]:
    """Each example: {"id", "context", "question", "answer"}."""
    raw = load_dataset(DATASET_REPO, split=split)
    examples = []
    for row in raw:
        if not row["answers"]["text"]:
            continue
        examples.append(
            {
                "id": row["id"],
                "context": row["context"],
                "question": row["question"],
                "answer": row["answers"]["text"][0],
            }
        )
    return examples


def sample_needles_and_haystack(
    examples: list[dict], needle_count: int, seed: int = 0
) -> tuple[list[dict], list[str]]:
    """Splits `examples` into `needle_count` needles (kept whole: context,
    question, and answer) and a shared pool of distractor paragraphs
    (every other example's context, its own question and answer
    discarded) that every needle draws its haystack filler from. Needles
    and distractors never share a context, so a distractor paragraph
    can't accidentally also answer a needle's question."""
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    needles = shuffled[:needle_count]
    distractor_pool = [ex["context"] for ex in shuffled[needle_count:]]
    return needles, distractor_pool


def build_document(needle_context: str, distractor_pool: list[str], total_tokens: int, position: str, seed: int) -> str:
    """Builds one haystack document of approximately `total_tokens` words
    (whitespace-split, same approximation the rag-architectures
    experiment's chunking module uses), with `needle_context` placed at
    the start, middle, or end. Distractor paragraphs are drawn from
    `distractor_pool`, shuffled per document, repeating through the pool
    if the target length needs more paragraphs than the pool has."""
    rng = random.Random(seed)
    needle_tokens = len(needle_context.split())
    filler_budget = max(total_tokens - needle_tokens, 0)

    pool = distractor_pool[:]
    rng.shuffle(pool)
    filler: list[str] = []
    filler_tokens = 0
    i = 0
    while filler_tokens < filler_budget:
        paragraph = pool[i % len(pool)]
        filler.append(paragraph)
        filler_tokens += len(paragraph.split())
        i += 1

    if position == "start":
        paragraphs = [needle_context] + filler
    elif position == "end":
        paragraphs = filler + [needle_context]
    else:  # "middle"
        midpoint = len(filler) // 2
        paragraphs = filler[:midpoint] + [needle_context] + filler[midpoint:]

    return "\n\n".join(paragraphs)
