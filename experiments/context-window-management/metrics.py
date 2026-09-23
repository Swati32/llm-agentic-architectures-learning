"""Turns raw per-document run records into the accuracy and operational
metrics this experiment compares context-management techniques on,
broken out by document length and needle position. That breakdown is the
entire point of this experiment: one aggregate accuracy number per
technique can't show *where* a technique starts failing, only that it
eventually does.
"""

import string
from collections import Counter

import pandas as pd

ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = [word for word in text.split() if word not in ARTICLES]
    return " ".join(words)


def exact_match(predicted: str | None, gold: str) -> float:
    if predicted is None:
        return 0.0
    return float(normalize_answer(predicted) == normalize_answer(gold))


def token_f1(predicted: str | None, gold: str) -> float:
    if predicted is None:
        return 0.0
    predicted_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    overlap = Counter(predicted_tokens) & Counter(gold_tokens)
    shared = sum(overlap.values())
    if shared == 0:
        return 0.0
    precision = shared / len(predicted_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def quality_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for technique, group in records.groupby("technique"):
        rows.append(
            {
                "technique": technique,
                "exact_match": group["exact_match"].mean(),
                "f1": group["f1"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("technique")


def quality_by_length(records: pd.DataFrame) -> pd.DataFrame:
    """One row per technique, one column per document length: mean F1.
    Isolates the pure length effect (averaged across all 3 positions)."""
    return records.pivot_table(index="technique", columns="length", values="f1", aggfunc="mean")


def quality_by_position(records: pd.DataFrame) -> pd.DataFrame:
    """One row per technique, one column per needle position: mean F1.
    Isolates the pure position effect (averaged across all 3 lengths) —
    this is the table that directly shows (or fails to show) lost in the
    middle for each technique."""
    return records.pivot_table(index="technique", columns="position", values="f1", aggfunc="mean")


def quality_by_length_and_position(records: pd.DataFrame) -> pd.DataFrame:
    """Long-form: one row per (technique, length, position), for the
    dashboard's heatmap view — the full grid neither of the two
    single-variable breakdowns above can show on its own."""
    return records.groupby(["technique", "length", "position"])["f1"].mean().reset_index()


def operational_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Time to first token uses the median, not the mean, deliberately:
    Ollama's self-reported prompt_eval_duration (what TTFT is built from)
    can spike when the host machine is under real memory pressure. A few
    such spikes would swamp a mean built from a few hundred samples; the
    median is unaffected by them."""
    rows = []
    for technique, group in records.groupby("technique"):
        rows.append(
            {
                "technique": technique,
                "mean_llm_calls": group["llm_calls"].mean(),
                "mean_prompt_tokens": group["total_prompt_tokens"].mean(),
                "mean_completion_tokens": group["total_completion_tokens"].mean(),
                "mean_latency_seconds": group["total_latency_seconds"].mean(),
                "median_time_to_first_token_seconds": group["time_to_first_token_seconds"].median(),
                "mean_context_payload_bytes": group["mean_context_payload_bytes"].mean(),
                "mean_wall_clock_seconds": group["wall_clock_seconds"].mean(),
                "error_rate": group["had_error"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("technique")
