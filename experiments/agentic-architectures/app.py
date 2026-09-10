"""Streamlit dashboard: methodology, cross-architecture comparison (with an
auto-generated what-worked/what-didn't overview), a per-architecture deep
dive (verbatim prompts + a full step-by-step trace), and a question
explorer for seeing every architecture's answer to the same question side
by side.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from metrics import agentic_summary, operational_summary, quality_summary
from techniques import ARCHITECTURES

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"

METRIC_LABELS = {
    "exact_match": "Exact Match",
    "f1": "F1",
    "bridge_f1": "Bridge-question F1",
    "comparison_f1": "Comparison-question F1",
    "mean_step_count": "Mean Step Count",
    "mean_tool_calls": "Mean Tool Calls",
    "tool_error_rate": "Tool Error Rate",
    "mean_handoffs": "Mean Handoffs",
    "early_termination_rate": "Early Termination Rate",
    "mean_state_overhead_bytes": "Mean State Overhead (bytes)",
    "mean_wall_clock_seconds": "Mean Wall-Clock Time (s)",
    "mean_prompt_tokens": "Mean Prompt Tokens",
    "mean_completion_tokens": "Mean Completion Tokens",
    "mean_latency_seconds": "Mean Model Latency (s)",
    "mean_time_to_first_token_seconds": "Mean Time to First Token (s)",
    "mean_context_payload_bytes": "Mean Context Payload (bytes)",
    "empty_retrieval_rate": "Empty (non-gold) Retrieval Rate",
    "error_rate": "Model Call Error Rate",
}
METRIC_FORMATS = {
    "exact_match": "{:.1%}", "f1": "{:.1%}", "bridge_f1": "{:.1%}", "comparison_f1": "{:.1%}",
    "tool_error_rate": "{:.1%}", "early_termination_rate": "{:.1%}", "empty_retrieval_rate": "{:.1%}",
    "error_rate": "{:.1%}",
    "mean_step_count": "{:.1f}", "mean_tool_calls": "{:.1f}", "mean_handoffs": "{:.1f}",
    "mean_wall_clock_seconds": "{:.1f}", "mean_latency_seconds": "{:.1f}",
    "mean_time_to_first_token_seconds": "{:.2f}",
    "mean_prompt_tokens": "{:.0f}", "mean_completion_tokens": "{:.0f}",
    "mean_state_overhead_bytes": "{:.0f}", "mean_context_payload_bytes": "{:.0f}",
}
# "lower" metrics are costs/failure modes where the smallest value wins;
# everything else is a quality metric where the largest value wins.
LOWER_IS_BETTER = {
    "mean_step_count", "mean_tool_calls", "tool_error_rate", "mean_handoffs",
    "early_termination_rate", "mean_state_overhead_bytes", "mean_wall_clock_seconds",
    "mean_prompt_tokens", "mean_completion_tokens", "mean_latency_seconds",
    "mean_time_to_first_token_seconds", "mean_context_payload_bytes",
    "empty_retrieval_rate", "error_rate",
}

st.set_page_config(page_title="Agentic Architectures Compared", layout="wide")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame]:
    payload = json.loads(RESULTS_PATH.read_text())
    return payload["architectures"], pd.DataFrame(payload["runs"])


def combined_metrics(records: pd.DataFrame) -> pd.DataFrame:
    return quality_summary(records).join(agentic_summary(records)).join(operational_summary(records))


def format_metric(metric: str, value: float) -> str:
    return METRIC_FORMATS.get(metric, "{:.2f}").format(value)


def ranked_metrics(combined: pd.DataFrame, architecture_key: str) -> list[tuple[str, float, int, int]]:
    """Every metric this architecture has, ranked 1 (best) to n (worst)
    against the other architectures, best-to-worst position first."""
    ranked = []
    for metric in combined.columns:
        column = combined[metric].dropna()
        if architecture_key not in column.index or len(column) < 2:
            continue
        ascending = metric in LOWER_IS_BETTER
        rank = int(column.rank(ascending=ascending, method="min")[architecture_key])
        ranked.append((metric, column[architecture_key], rank, len(column)))
    ranked.sort(key=lambda item: item[2])
    return ranked


def notable_metrics(combined: pd.DataFrame, architecture_key: str, side: str, count: int = 2) -> list[tuple[str, float, int, int]]:
    """The 2 metrics that actually distinguish this architecture, not just
    whichever happens to sort first. A metric every architecture is tied on
    (e.g. a 0% error rate across the board) says nothing about this specific
    architecture, so it's excluded entirely; and a "struggled" pick has to
    land in the bottom half, not just be whatever ranked worst among an
    otherwise strong scorecard."""
    n = len(combined.index)
    picks = []
    for metric, value, rank, count_n in ranked_metrics(combined, architecture_key):
        column = combined[metric].dropna()
        if column.max() == column.min():
            continue  # no variance across architectures, not a real distinction
        if side == "best" and rank <= 2:
            picks.append((metric, value, rank, count_n))
        elif side == "worst" and rank >= count_n - 1:
            picks.append((metric, value, rank, count_n))
    if side == "worst":
        picks.sort(key=lambda item: -item[2])
    return picks[:count]


def render_trace(steps: list[dict]) -> None:
    for i, step in enumerate(steps, start=1):
        if step["kind"] == "tool_call":
            if step["error"]:
                marker = f"error: {step['error']}"
            elif step["is_gold_retrieval"]:
                marker = "✅ hit a gold paragraph"
            else:
                marker = "❌ hit a distractor paragraph"
            st.markdown(f"**{i}. [{step['role']}] search(`{step['detail']}`)** — {marker}, {step['latency_seconds']*1000:.0f} ms")
            if step["output"]:
                st.caption(step["output"][:280] + ("…" if len(step["output"]) > 280 else ""))
        else:
            cost = f"{step['latency_seconds']:.1f}s, {step['prompt_tokens']}+{step['completion_tokens']} tokens"
            st.markdown(f"**{i}. [{step['role']}] model call** — {step['detail']} ({cost})")
            if step["error"]:
                st.caption(f"Error: {step['error']}")
            elif step["output"]:
                st.text(step["output"])


st.markdown(
    """
    <div style="background:linear-gradient(135deg, #0F766E, #1D4ED8);padding:1.5rem 1.5rem;border-radius:0.5rem;margin-bottom:1.5rem;">
        <h1 style="color:white;margin:0;font-size:2rem;">Agentic Architectures: Sequential vs. Orchestrator Styles</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not RESULTS_PATH.exists():
    st.warning("No results yet. Run `python3 run_experiment.py` in this experiment's folder first. This dashboard reads from `results/records.json`.")
    st.stop()

architectures_meta, records = load_results()
combined = combined_metrics(records)

tab_methodology, tab_comparison, tab_deep_dive, tab_explorer = st.tabs(
    ["Methodology", "Architecture Comparison", "Architecture Deep-Dive", "Question Explorer"]
)

with tab_methodology:
    st.header("Goal")
    st.markdown(
        "Compare architectural styles for multi-step agent tasks, fixed sequential "
        "pipelines versus adaptive orchestrator and supervisor patterns, on the "
        "same problem, to see which fails where and what it costs operationally "
        "(steps, latency, tool errors, coordination overhead) to get there."
    )

    st.header("The problem: multi-hop question answering")
    st.markdown(
        f"**[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)** "
        "(distractor config): each question needs facts from two different "
        "paragraphs to answer. Every question ships with 10 candidate paragraphs, "
        "2 that actually support the answer and 8 unrelated distractors. That "
        "paragraph set is the fixed corpus a `search()` tool retrieves over, "
        "using a small local sentence-transformer "
        "(`all-MiniLM-L6-v2`) for semantic similarity, not live web search — every "
        "architecture is judged against exactly the same evidence.\n\n"
        f"Every architecture answers the same **{records['question_id'].nunique()} "
        "sampled questions**, split evenly between the two question types HotpotQA "
        "labels: **bridge** (hop 2 needs an entity hop 1 finds) and **comparison** "
        "(two independent lookups compared against each other). The `search()` tool "
        "returns only its single best-matching paragraph (k=1) — with 2 gold "
        "paragraphs needed, one search can never fully answer a question, forcing "
        "every architecture to actually decide whether and how to search again."
    )

    st.header("Architectures compared")
    for key, meta in architectures_meta.items():
        st.markdown(f"**{meta['name']}** — {meta['description']}")

    st.header("Agentic metrics")
    st.markdown(
        "- **Step / loop count** — how many model or tool calls one question took\n"
        "- **Tool execution latency** — wall-clock time spent in `search()` itself\n"
        "- **Tool error / retry rate** — fraction of runs with at least one failed tool call\n"
        "- **Inter-agent handoff count** — how many times control passed between roles "
        "(planner, worker, verifier, ...); 0 for the single-agent baseline by definition\n"
        "- **Early termination rate** — fraction of runs that hit their step cap without "
        "a clean finish/verdict, rather than stopping because the architecture decided it was done\n"
        "- **State overhead** — size (bytes) of the shared state passed at the final handoff; "
        "for the single agent this is its whole running transcript instead"
    )

    st.header("Operational metrics")
    st.markdown(
        "- **Tokens used**, **context payload size** — cost drivers, and directly comparable here "
        "since every architecture runs on the same local model\n"
        "- **Latency** and **time to first token** — of the *first* model call in the run\n"
        "- **Empty retrieval rate** — redefined for this task: since `search()` always returns its "
        "top-1 match, \"empty\" here means the retrieved paragraph wasn't one of the 2 gold "
        "paragraphs, i.e. a wasted or misleading lookup\n"
        "- **Error rate** — fraction of model calls that failed outright"
    )
    st.caption(
        "Semantic cache hit rate isn't tracked in this experiment: nothing here caches across "
        "questions, so the metric wouldn't be meaningful."
    )

    st.header("Model backend")
    st.markdown(
        "Every model call in every architecture goes through the same local "
        "[Ollama](https://ollama.com) server running `llama3.1:8b` (see `llm_client.py`), "
        "so differences in the results come from architecture, not from which model answered."
    )

with tab_comparison:
    st.header("What worked, what didn't")
    st.caption(
        "For each architecture, its top 2 metrics that actually distinguish it from the other "
        "four (What worked) and its bottom 2 (Struggled). A metric every architecture is tied "
        "on, e.g. a 0% error rate across the board, is excluded: it isn't a distinguishing "
        "strength or weakness for any one of them."
    )

    def describe_side(key: str, side: str) -> tuple[str, str]:
        picks = notable_metrics(combined, key, side)
        labels = [f"{METRIC_LABELS[m]}: {format_metric(m, v)} (rank {r}/{n})" for m, v, r, n in picks]
        while len(labels) < 2:
            labels.append("Nothing standout in this sample")
        return labels[0], labels[1]

    overview_rows = []
    for key, meta in architectures_meta.items():
        best_1, best_2 = describe_side(key, "best")
        worst_1, worst_2 = describe_side(key, "worst")
        overview_rows.append(
            {"Architecture": meta["name"], "Worked well": best_1, "Also strong": best_2, "Struggled": worst_1, "Also weak": worst_2}
        )
    st.dataframe(pd.DataFrame(overview_rows).set_index("Architecture"), width="stretch")

    st.header("Quality: all architectures")
    quality = quality_summary(records)
    st.dataframe(quality.rename(columns=METRIC_LABELS).style.format("{:.1%}"), width="stretch")
    st.bar_chart(quality[["exact_match", "f1", "bridge_f1", "comparison_f1"]])

    st.header("Agentic cost: all architectures")
    agentic = agentic_summary(records)
    agentic_formats = {METRIC_LABELS[c]: METRIC_FORMATS.get(c, "{:.2f}") for c in agentic.columns}
    st.dataframe(agentic.rename(columns=METRIC_LABELS).style.format(agentic_formats), width="stretch")
    st.bar_chart(agentic[["mean_step_count", "mean_handoffs"]])
    st.bar_chart(agentic[["early_termination_rate", "tool_error_rate"]])

    st.header("Operational cost: all architectures")
    operational = operational_summary(records)
    operational_formats = {METRIC_LABELS[c]: METRIC_FORMATS.get(c, "{:.2f}") for c in operational.columns}
    st.dataframe(operational.rename(columns=METRIC_LABELS).style.format(operational_formats), width="stretch")
    st.bar_chart(operational[["mean_latency_seconds"]])
    st.bar_chart(operational[["mean_prompt_tokens", "mean_completion_tokens"]])

    st.header("Accuracy vs. wall-clock time")
    tradeoff = quality[["exact_match"]].join(agentic[["mean_wall_clock_seconds"]])
    st.scatter_chart(tradeoff, x="mean_wall_clock_seconds", y="exact_match")

with tab_deep_dive:
    architecture_key = st.selectbox(
        "Architecture", options=list(ARCHITECTURES.keys()), format_func=lambda k: architectures_meta[k]["name"]
    )
    meta = architectures_meta[architecture_key]
    st.subheader(meta["name"])
    st.markdown(meta["description"])

    st.markdown("**How it ranks against the other four architectures, best to worst:**")
    ranked = ranked_metrics(combined, architecture_key)
    ranked_table = pd.DataFrame(
        [{"Metric": METRIC_LABELS[m], "Value": format_metric(m, v), "Rank": f"{r} of {n}"} for m, v, r, n in ranked]
    )
    st.dataframe(ranked_table.set_index("Metric"), width="stretch")

    st.markdown("**Prompts used, verbatim:**")
    for role, prompt in meta["prompts"].items():
        with st.expander(f"{role}"):
            st.text(prompt)

    st.markdown("**Example run, full trace:**")
    architecture_records = records[records["architecture"] == architecture_key]
    question_options = architecture_records["question"].tolist()
    example_question = st.selectbox("Question", options=question_options, key=f"trace_{architecture_key}")
    example_run = architecture_records[architecture_records["question"] == example_question].iloc[0]
    st.markdown(f"**Gold answer:** `{example_run['gold_answer']}` · **Predicted:** `{example_run['predicted_answer']}` · **F1:** {example_run['f1']:.2f}")
    render_trace(example_run["steps"])

with tab_explorer:
    question_text = st.selectbox("Question", options=sorted(records["question"].unique()))
    question_records = records[records["question"] == question_text].set_index("architecture")

    gold_answer = question_records.iloc[0]["gold_answer"]
    question_type = question_records.iloc[0]["question_type"]
    st.markdown(f"**Gold answer:** `{gold_answer}` · **Type:** `{question_type}`")

    summary_columns = ["predicted_answer", "exact_match", "f1", "step_count", "tool_calls", "handoffs", "wall_clock_seconds"]
    display_summary = question_records[summary_columns].rename(index=lambda k: architectures_meta[k]["name"])
    st.dataframe(display_summary, width="stretch")

    architecture_key = st.selectbox(
        "Inspect one architecture's full trace for this question",
        options=list(ARCHITECTURES.keys()),
        format_func=lambda k: architectures_meta[k]["name"],
    )
    render_trace(question_records.loc[architecture_key, "steps"])
