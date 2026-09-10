"""Turns raw per-question run records into the quality, agentic, and
operational metrics this experiment compares architectures on. See
CLAUDE.md for the metric definitions."""

import re
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
    """Standard SQuAD-style token-overlap F1 — HotpotQA answers are short
    spans or yes/no, so partial credit for getting some but not all of a
    multi-word span matters more here than plain exact match."""
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
    for architecture, group in records.groupby("architecture"):
        rows.append(
            {
                "architecture": architecture,
                "exact_match": group["exact_match"].mean(),
                "f1": group["f1"].mean(),
                "bridge_f1": group.loc[group["question_type"] == "bridge", "f1"].mean(),
                "comparison_f1": group.loc[group["question_type"] == "comparison", "f1"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("architecture")


def agentic_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for architecture, group in records.groupby("architecture"):
        rows.append(
            {
                "architecture": architecture,
                "mean_step_count": group["step_count"].mean(),
                "mean_tool_calls": group["tool_calls"].mean(),
                "tool_error_rate": (group["tool_errors"] > 0).mean(),
                "mean_handoffs": group["handoffs"].mean(),
                "early_termination_rate": group["early_terminated"].mean(),
                "mean_state_overhead_bytes": group["state_overhead_bytes"].mean(),
                "mean_wall_clock_seconds": group["wall_clock_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("architecture")


def operational_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for architecture, group in records.groupby("architecture"):
        rows.append(
            {
                "architecture": architecture,
                "mean_prompt_tokens": group["total_prompt_tokens"].mean(),
                "mean_completion_tokens": group["total_completion_tokens"].mean(),
                "mean_latency_seconds": group["total_latency_seconds"].mean(),
                "mean_time_to_first_token_seconds": group["time_to_first_token_seconds"].mean(),
                "mean_context_payload_bytes": group["mean_context_payload_bytes"].mean(),
                "empty_retrieval_rate": group["empty_retrieval_rate"].mean(),
                "error_rate": group["had_error"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("architecture")
