"""Turns raw per-query run records into the retrieval, generation, and
operational metrics this experiment compares architectures on.

Two things are deliberately *not* here. First, semantic cache hit rate:
no architecture in this experiment caches anything, so the metric would
read 0% everywhere and say nothing about architecture, only about the
absence of a caching layer we didn't build. Second, an LLM-judged
faithfulness/answer-relevance score in the style of RAGAS
(https://arxiv.org/abs/2309.15217): that needs a judge call per record,
which roughly doubles this experiment's already-large local LLM call
count. `faithfulness_heuristic()` below is a lexical-overlap stand-in
instead — see the README's "what this task does and doesn't test" section
for what that approximation misses.
"""

import re
import string
from collections import Counter

import pandas as pd

ARTICLES = {"a", "an", "the"}
STOPWORDS = ARTICLES | {
    "is", "was", "were", "are", "be", "been", "of", "in", "on", "at", "to", "for",
    "and", "or", "with", "as", "by", "that", "this", "it", "its", "their", "his", "her",
}


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


def faithfulness_heuristic(answer: str | None, context_text: str) -> float | None:
    """Approximates groundedness without a judge LLM call: what fraction
    of the answer's own content words (stopwords and articles stripped)
    also appear somewhere in the retrieved context. A high score doesn't
    prove every claim in the answer is actually supported, two unrelated
    sentences can share vocabulary, only that the answer isn't reaching
    for words the context never used. Returns None for an abstained
    answer, since 'Insufficient information.' makes no claim to check."""
    if answer is None:
        return None
    normalized = normalize_answer(answer)
    if normalized == "insufficient information":
        return None
    answer_words = [w for w in normalized.split() if w not in STOPWORDS]
    if not answer_words:
        return None
    context_words = set(normalize_answer(context_text).split())
    grounded = sum(1 for w in answer_words if w in context_words)
    return grounded / len(answer_words)


def retrieval_recall_at_k(is_gold_flags: list[bool]) -> float:
    return float(any(is_gold_flags))


def retrieval_precision_at_k(is_gold_flags: list[bool]) -> float:
    return sum(is_gold_flags) / len(is_gold_flags) if is_gold_flags else 0.0


def retrieval_mrr(is_gold_flags: list[bool]) -> float:
    for rank, is_gold in enumerate(is_gold_flags, start=1):
        if is_gold:
            return 1.0 / rank
    return 0.0


def quality_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Answerable-query correctness (exact match, F1) and null-query
    abstention behavior, side by side. A high F1 that's actually driven by
    hallucinating on null queries the same way it answers real ones would
    be invisible in F1 alone, hence tracking the two behaviors separately
    rather than one blended accuracy number."""
    answerable = records[records["question_type"] != "null_query"]
    null_queries = records[records["question_type"] == "null_query"]
    rows = []
    for architecture in records["architecture"].unique():
        ans_group = answerable[answerable["architecture"] == architecture]
        null_group = null_queries[null_queries["architecture"] == architecture]
        rows.append(
            {
                "architecture": architecture,
                "exact_match": ans_group["exact_match"].mean(),
                "f1": ans_group["f1"].mean(),
                "incorrect_abstention_rate": ans_group["abstained"].mean(),
                "correct_abstention_rate": null_group["abstained"].mean() if len(null_group) else None,
                "faithfulness": ans_group["faithfulness"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("architecture")


def quality_by_question_type(records: pd.DataFrame) -> pd.DataFrame:
    """The 'which architecture wins at which scenario' table: F1 for the
    3 answerable types, correct-abstention rate for null queries, one row
    per architecture."""
    rows = []
    for architecture, group in records.groupby("architecture"):
        row = {"architecture": architecture}
        for qtype in ["inference_query", "comparison_query", "temporal_query"]:
            subset = group[group["question_type"] == qtype]
            row[f"{qtype}_f1"] = subset["f1"].mean() if len(subset) else None
        null_subset = group[group["question_type"] == "null_query"]
        row["null_query_abstention_rate"] = null_subset["abstained"].mean() if len(null_subset) else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("architecture")


def retrieval_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Retrieval-quality metrics, computed only over answerable queries
    (null queries have no gold chunk by construction, so recall/precision/
    MRR are undefined for them, not 0)."""
    answerable = records[records["question_type"] != "null_query"]
    rows = []
    for architecture, group in answerable.groupby("architecture"):
        rows.append(
            {
                "architecture": architecture,
                "recall_at_k": group["recall_at_k"].mean(),
                "precision_at_k": group["precision_at_k"].mean(),
                "mrr": group["mrr"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("architecture")


def operational_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Time to first token uses the median, not the mean, deliberately.
    Ollama's self-reported `prompt_eval_duration` (what TTFT is built from)
    can spike into the hundreds of seconds when the host machine is under
    real memory pressure, sometimes reporting a TTFT *larger than the
    call's own total latency*, which is impossible and is a serving
    artifact, not a property of the query. A few such spikes would swamp a
    mean built from 80 samples; the median is unaffected by them. See the
    README's caveats section for when this showed up in this experiment's
    own run."""
    rows = []
    for architecture, group in records.groupby("architecture"):
        rows.append(
            {
                "architecture": architecture,
                "mean_llm_calls": group["llm_calls"].mean(),
                "mean_retrieval_rounds": group["retrieval_rounds"].mean(),
                "mean_prompt_tokens": group["total_prompt_tokens"].mean(),
                "mean_completion_tokens": group["total_completion_tokens"].mean(),
                "mean_latency_seconds": group["total_latency_seconds"].mean(),
                "median_time_to_first_token_seconds": group["time_to_first_token_seconds"].median(),
                "mean_context_payload_bytes": group["mean_context_payload_bytes"].mean(),
                "mean_wall_clock_seconds": group["wall_clock_seconds"].mean(),
                "empty_retrieval_rate": group["empty_retrieval_rate"].mean(),
                "error_rate": group["had_error"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("architecture")
