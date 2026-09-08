"""Turns raw per-query records into the quality and operational metrics
this experiment compares techniques on. See CLAUDE.md for definitions."""

import pandas as pd
from sklearn.metrics import f1_score

from taxonomy import INTENT_TO_GROUP

ALL_INTENTS = sorted(INTENT_TO_GROUP)
UNRESOLVED_LABEL = "<no valid label>"  # sentinel so an invalid prediction always counts as wrong


def quality_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for technique, group in records.groupby("technique"):
        predicted_intents = group["predicted_intent"].fillna(UNRESOLVED_LABEL)
        rows.append(
            {
                "technique": technique,
                "fine_accuracy": (group["predicted_intent"] == group["gold_intent"]).mean(),
                "macro_f1": f1_score(
                    group["gold_intent"],
                    predicted_intents,
                    labels=ALL_INTENTS,
                    average="macro",
                    zero_division=0,
                ),
                "coarse_accuracy": (group["predicted_group"] == group["gold_group"]).mean(),
                "label_validity_rate": group["is_valid"].mean(),
                "hierarchy_consistency_rate": group["is_hierarchy_consistent"].mean(),
                "error_rate": group["had_error"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("technique")


def operational_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for technique, group in records.groupby("technique"):
        rows.append(
            {
                "technique": technique,
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_completion_tokens": group["completion_tokens"].mean(),
                "mean_latency_seconds": group["latency_seconds"].mean(),
                "mean_time_to_first_token_seconds": group["time_to_first_token_seconds"].mean(),
                "mean_context_payload_bytes": group["context_payload_bytes"].mean(),
                "semantic_cache_hit_rate": group["cache_hit"].mean(),
                "empty_retrieval_rate": group["retrieval_empty"].mean(),
                "error_rate": group["had_error"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("technique")


def confusion_pairs(records: pd.DataFrame, technique: str) -> pd.DataFrame:
    technique_records = records[records["technique"] == technique]
    predicted = technique_records["predicted_intent"].fillna(UNRESOLVED_LABEL)
    return pd.crosstab(technique_records["gold_intent"], predicted)
