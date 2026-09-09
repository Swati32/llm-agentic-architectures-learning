"""Streamlit dashboard: methodology, cross-technique comparison, a
per-technique deep-dive (exact prompts + confusion matrix), and a query
explorer for seeing every technique's answer to the same question side by side.
"""

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from metrics import confusion_pairs, operational_summary, quality_summary
from taxonomy import COARSE_GROUPS
from techniques import TECHNIQUES

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"

QUALITY_METRIC_LABELS = {
    "fine_accuracy": "Fine Accuracy",
    "macro_f1": "Macro F1",
    "coarse_accuracy": "Coarse Accuracy",
    "label_validity_rate": "Label Validity Rate",
    "hierarchy_consistency_rate": "Hierarchy Consistency Rate",
    "error_rate": "Error Rate",
}

OPERATIONAL_METRIC_LABELS = {
    "mean_prompt_tokens": "Mean Prompt Tokens",
    "mean_completion_tokens": "Mean Completion Tokens",
    "mean_latency_seconds": "Mean Latency (s)",
    "mean_time_to_first_token_seconds": "Mean Time to First Token (s)",
    "mean_context_payload_bytes": "Mean Context Payload (bytes)",
    "semantic_cache_hit_rate": "Semantic Cache Hit Rate",
    "empty_retrieval_rate": "Empty Retrieval Rate",
    "error_rate": "Error Rate",
}

st.set_page_config(page_title="Intent Classification: Prompting Techniques", layout="wide")


@st.cache_data
def load_records() -> pd.DataFrame:
    records = json.loads(RESULTS_PATH.read_text())
    return pd.DataFrame(records)


CATALOG_LINE = re.compile(r"^ {0,4}- .+$")
CATALOG_MIN_LINES_TO_COLLAPSE = 12  # short lists (e.g. the 10 group names) stay as-is


def collapse_label_catalog(content: str) -> str:
    """The full 77-intent catalog repeats in almost every prompt and drowns
    out what's actually distinctive about each technique. Collapse any long
    run of catalog-style bullet lines into a one-line placeholder."""
    lines = content.split("\n")
    collapsed = []
    i = 0
    while i < len(lines):
        if CATALOG_LINE.match(lines[i]):
            start = i
            while i < len(lines) and CATALOG_LINE.match(lines[i]):
                i += 1
            block_length = i - start
            if block_length >= CATALOG_MIN_LINES_TO_COLLAPSE:
                collapsed.append(f"[... {block_length} lines of the label catalog omitted, see taxonomy.py ...]")
            else:
                collapsed.extend(lines[start:i])
        else:
            collapsed.append(lines[i])
            i += 1
    return "\n".join(collapsed)


def render_messages(messages: list[dict]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.text(collapse_label_catalog(message["content"]))


def render_prompt_calls(prompts_used: list[list[dict]]) -> None:
    if len(prompts_used) == 1:
        render_messages(prompts_used[0])
        return
    for call_index, messages in enumerate(prompts_used, start=1):
        st.caption(f"Call {call_index} of {len(prompts_used)}")
        render_messages(messages)


st.markdown(
    """
    <div style="background:linear-gradient(135deg, #7C3AED, #C4B5FD);padding:1.5rem 1.5rem;border-radius:0.5rem;margin-bottom:1.5rem;">
        <h1 style="color:white;margin:0;font-size:2rem;">Intent Classification: Comparing Prompting Techniques</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not RESULTS_PATH.exists():
    st.warning(
        "No results yet. Run `python run_experiment.py` in this experiment's "
        "folder first. This dashboard reads from `results/records.json`."
    )
    st.stop()

records = load_records()
quality = quality_summary(records)
operational = operational_summary(records)

tab_methodology, tab_comparison, tab_deep_dive, tab_explorer = st.tabs(
    ["Methodology", "Technique Comparison", "Technique Deep-Dive", "Query Explorer"]
)

with tab_methodology:
    st.header("Goal")
    st.markdown(
        "Learn how different prompting techniques affect **both** classification "
        "quality and operational cost on the same task, and build intuition for "
        "when each technique's cost is actually worth paying."
    )

    st.header("Dataset")
    st.markdown(
        f"**[Banking77](https://huggingface.co/datasets/PolyAI/banking77)** — 77 "
        "fine-grained banking intents, chosen because its classes are "
        "semantically close (e.g. `card_not_working` vs `virtual_card_not_working`), "
        "making it a real stress test for prompt wording rather than a task any "
        "technique trivially solves.\n\n"
        f"Every technique is evaluated on the same **stratified sample of "
        f"{records['text'].nunique()} test queries**, proportional to each intent's "
        "share of the test set, so no technique is judged on an easier or harder slice."
    )

    st.header("Output schema: coarse group + intent")
    st.markdown(
        "Banking77 ships no official coarse categories, so this project defines its "
        "own 10-group taxonomy over the 77 intents (see `taxonomy.py`) — it becomes "
        "ground truth for the Coarse Accuracy metric below. Every technique is asked "
        "for **both** a coarse group and a fine intent, which enables:\n\n"
        "- **Coarse Accuracy** — did the predicted group match?\n"
        "- **Fine Accuracy** — did the exact predicted intent match?\n"
        "- **Hierarchy Consistency Rate** — does the predicted intent actually belong "
        "to the predicted group, per this taxonomy? (catches a self-contradictory answer, "
        "independent of whether either field is correct)"
    )
    with st.expander("View the full 10-group taxonomy"):
        for group, intents in COARSE_GROUPS.items():
            st.markdown(f"**{group}** ({len(intents)}): {', '.join(intents)}")

    st.header("Techniques compared")
    for technique in TECHNIQUES.values():
        st.markdown(f"**{technique.NAME}** — {technique.DESCRIPTION}")

    st.header("Quality metrics")
    st.markdown(
        "- **Fine Accuracy** — exact match between predicted and gold intent\n"
        "- **Macro F1** — unweighted mean F1 across all 77 intents, so rare classes "
        "aren't drowned out by common ones\n"
        "- **Coarse Accuracy** — exact match between predicted and gold coarse group\n"
        "- **Label Validity Rate** — fraction of responses that named a real intent "
        "label at all, separate from whether it was the *correct* one\n"
        "- **Hierarchy Consistency Rate** — see above"
    )

    st.header("Operational metrics")
    st.markdown(
        "- **Tokens used** (prompt + completion) and **Context Payload Size** — cost drivers\n"
        "- **Latency** and **Time to First Token** — user-facing responsiveness\n"
        "- **Error Rate** — fraction of calls that failed outright\n"
        "- **Semantic Cache Hit Rate** — fraction of queries answered from a cached "
        "prediction for a near-duplicate query, without calling the model again "
        "(applies to the semantic-retrieval technique only)\n"
        "- **Empty Retrieval Rate** — fraction of queries for which no training "
        "example was similar enough to use as a demonstration (also retrieval-only)"
    )

with tab_comparison:
    st.header("Quality: all techniques")
    st.dataframe(quality.rename(columns=QUALITY_METRIC_LABELS).style.format("{:.1%}"))
    st.bar_chart(quality[["fine_accuracy", "coarse_accuracy", "macro_f1"]])

    st.header("Operational cost: all techniques")
    st.dataframe(
        operational.rename(columns=OPERATIONAL_METRIC_LABELS).style.format(
            {
                "Mean Prompt Tokens": "{:.0f}",
                "Mean Completion Tokens": "{:.0f}",
                "Mean Latency (s)": "{:.2f}",
                "Mean Time to First Token (s)": "{:.2f}",
                "Mean Context Payload (bytes)": "{:.0f}",
                "Semantic Cache Hit Rate": "{:.1%}",
                "Empty Retrieval Rate": "{:.1%}",
                "Error Rate": "{:.1%}",
            }
        )
    )
    st.bar_chart(operational[["mean_latency_seconds"]])
    st.bar_chart(operational[["mean_prompt_tokens", "mean_completion_tokens"]])

    st.header("Accuracy vs. latency")
    tradeoff = quality[["fine_accuracy"]].join(operational[["mean_latency_seconds"]])
    st.scatter_chart(tradeoff, x="mean_latency_seconds", y="fine_accuracy")

with tab_deep_dive:
    technique_key = st.selectbox(
        "Technique", options=list(TECHNIQUES.keys()), format_func=lambda k: TECHNIQUES[k].NAME
    )
    technique_module = TECHNIQUES[technique_key]
    technique_records = records[records["technique"] == technique_key]

    st.subheader(technique_module.NAME)
    st.markdown(technique_module.DESCRIPTION)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Fine Accuracy", f"{quality.loc[technique_key, 'fine_accuracy']:.1%}")
    metric_columns[1].metric("Macro F1", f"{quality.loc[technique_key, 'macro_f1']:.1%}")
    metric_columns[2].metric("Mean Latency", f"{operational.loc[technique_key, 'mean_latency_seconds']:.2f}s")

    st.markdown("**Example prompt** (from the first query in the test sample):")
    render_prompt_calls(technique_records.iloc[0]["prompts_used"])

    st.markdown("**Top confusions** (gold intent vs. predicted intent, most frequent mismatches):")
    confusion = confusion_pairs(records, technique_key)
    mismatches = confusion.copy()
    for intent in mismatches.index:
        if intent in mismatches.columns:
            mismatches.loc[intent, intent] = 0
    top_confused_intents = mismatches.sum(axis=1).sort_values(ascending=False).head(10).index
    st.dataframe(confusion.loc[top_confused_intents])

    st.markdown("**Every query, this technique's result:**")
    display_columns = ["text", "gold_intent", "predicted_intent", "gold_group", "predicted_group", "is_valid"]
    st.dataframe(technique_records[display_columns], width='stretch')

with tab_explorer:
    query_text = st.selectbox("Query", options=sorted(records["text"].unique()))
    query_records = records[records["text"] == query_text].set_index("technique")

    gold_intent = query_records.iloc[0]["gold_intent"]
    gold_group = query_records.iloc[0]["gold_group"]
    st.markdown(f"**Gold answer:** `{gold_group}` / `{gold_intent}`")

    summary_columns = ["predicted_group", "predicted_intent", "latency_seconds", "prompt_tokens", "completion_tokens"]
    display_summary = query_records[summary_columns].rename(index=lambda k: TECHNIQUES[k].NAME)
    display_summary["correct"] = display_summary["predicted_intent"] == gold_intent
    st.dataframe(display_summary, width='stretch')

    technique_key = st.selectbox(
        "Inspect one technique's exact prompt and response for this query",
        options=list(TECHNIQUES.keys()),
        format_func=lambda k: TECHNIQUES[k].NAME,
    )
    record = query_records.loc[technique_key]
    render_prompt_calls(record["prompts_used"])
    st.markdown("**Raw response:**")
    st.text(record["raw_response"])
