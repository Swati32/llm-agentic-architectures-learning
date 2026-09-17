"""Streamlit dashboard: methodology, cross-shape comparison (with an
auto-generated what-worked/what-didn't overview), a per-shape deep dive
(verbatim prompts + a full step-by-step trace), and a question explorer for
seeing every shape's answer to the same question side by side.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from metrics import agentic_summary, operational_summary, quality_summary
from techniques import SHAPES

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
    "redundant_step_rate": "Redundant Step Rate",
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
    "tool_error_rate": "{:.1%}", "early_termination_rate": "{:.1%}", "redundant_step_rate": "{:.1%}",
    "empty_retrieval_rate": "{:.1%}", "error_rate": "{:.1%}",
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
    "early_termination_rate", "redundant_step_rate", "mean_state_overhead_bytes",
    "mean_wall_clock_seconds", "mean_prompt_tokens", "mean_completion_tokens",
    "mean_latency_seconds", "mean_time_to_first_token_seconds", "mean_context_payload_bytes",
    "empty_retrieval_rate", "error_rate",
}

st.set_page_config(page_title="Evaluation Criteria in Practice", layout="wide")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame]:
    payload = json.loads(RESULTS_PATH.read_text())
    return payload["shapes"], pd.DataFrame(payload["runs"])


def combined_metrics(records: pd.DataFrame) -> pd.DataFrame:
    return quality_summary(records).join(agentic_summary(records)).join(operational_summary(records))


def format_metric(metric: str, value: float) -> str:
    return METRIC_FORMATS.get(metric, "{:.2f}").format(value)


def format_predicted(value) -> str:
    return "(no answer)" if pd.isna(value) else str(value)


def ranked_metrics(combined: pd.DataFrame, shape_key: str) -> list[tuple[str, float, int, int]]:
    """Every metric this shape has, ranked 1 (best) to n (worst) against
    the other shapes, best-to-worst position first."""
    ranked = []
    for metric in combined.columns:
        column = combined[metric].dropna()
        if shape_key not in column.index or len(column) < 2:
            continue
        ascending = metric in LOWER_IS_BETTER
        rank = int(column.rank(ascending=ascending, method="min")[shape_key])
        ranked.append((metric, column[shape_key], rank, len(column)))
    ranked.sort(key=lambda item: item[2])
    return ranked


def notable_metrics(combined: pd.DataFrame, shape_key: str, side: str, count: int = 2) -> list[tuple[str, float, int, int]]:
    """The 2 metrics that actually distinguish this shape, not just
    whichever happens to sort first. A metric every shape is tied on (e.g.
    a 0% error rate across the board) says nothing about this specific
    shape, so it's excluded entirely."""
    picks = []
    for metric, value, rank, count_n in ranked_metrics(combined, shape_key):
        column = combined[metric].dropna()
        if column.max() == column.min():
            continue  # no variance across shapes, not a real distinction
        if side == "best" and rank <= 2:
            picks.append((metric, value, rank, count_n))
        elif side == "worst" and rank >= count_n - 1:
            picks.append((metric, value, rank, count_n))
    if side == "worst":
        picks.sort(key=lambda item: -item[2])
    return picks[:count]


def render_mermaid(diagram: str, height: int = 220) -> None:
    components.html(
        f"""
        <div class="mermaid" id="diagram" style="font-family: sans-serif;">{diagram}</div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: false, flowchart: {{ useMaxWidth: true }} }});
            function renderWhenReady(attemptsLeft) {{
                var el = document.getElementById("diagram");
                if (el.offsetWidth > 0 || attemptsLeft <= 0) {{
                    mermaid.run({{ nodes: [el] }});
                }} else {{
                    requestAnimationFrame(function () {{ renderWhenReady(attemptsLeft - 1); }});
                }}
            }}
            renderWhenReady(60);
        </script>
        """,
        height=height,
    )


def render_trace(steps: list[dict]) -> None:
    for i, step in enumerate(steps, start=1):
        if step["kind"] == "tool_call":
            if step["error"]:
                marker = f"error: {step['error']}"
            elif step["is_gold_retrieval"]:
                marker = "✅ hit a gold paragraph"
            else:
                marker = "❌ hit a distractor paragraph"
            st.markdown(f"**{i}. [{step['role']}] search(`{step['detail']}`):** {marker}, {step['latency_seconds']*1000:.0f} ms")
            if step["output"]:
                st.caption(step["output"][:280] + ("…" if len(step["output"]) > 280 else ""))
        else:
            cost = f"{step['latency_seconds']:.1f}s, {step['prompt_tokens']}+{step['completion_tokens']} tokens"
            st.markdown(f"**{i}. [{step['role']}] model call:** {step['detail']} ({cost})")
            if step["error"]:
                st.caption(f"Error: {step['error']}")
            elif step["output"]:
                st.text(step["output"])


st.markdown(
    """
    <div style="background:linear-gradient(135deg, #5B21B6, #DB2777);padding:1.5rem 1.5rem;border-radius:0.5rem;margin-bottom:1.5rem;">
        <h1 style="color:white;margin:0;font-size:2rem;">Evaluation Criteria in Practice: Plain LLM vs. RAG vs. Single Agent vs. Multi-Agent</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not RESULTS_PATH.exists():
    st.warning("No results yet. Run `python3 run_experiment.py` in this experiment's folder first. This dashboard reads from `results/records.json`.")
    st.stop()

shapes_meta, records = load_results()
combined = combined_metrics(records)

tab_methodology, tab_comparison, tab_deep_dive, tab_explorer = st.tabs(
    ["Methodology", "Shape Comparison", "Shape Deep-Dive", "Question Explorer"]
)

with tab_methodology:
    st.header("Goal")
    st.markdown(
        "The prior two experiments in this repo, "
        "[RAG Architectures](../rag-architectures/README.md) and "
        "[Agentic Architectures](../agentic-architectures/README.md), each compared "
        "*techniques within one shape* (which RAG design, which agent control flow). "
        "This experiment holds the task and dataset fixed and varies the *shape itself*: "
        "plain LLM call, RAG, single agent, multi-agent, to directly test the central claim "
        "of [EVALUATION_CRITERIA.md](../../EVALUATION_CRITERIA.md): that a task/quality "
        "metric alone under-determines outcomes more as a system's shape gains more decision "
        "points, and that the metric layer that actually explains a close call is whichever "
        "one is specific to what that shape can uniquely get wrong."
    )

    st.header("The problem: multi-hop question answering")
    st.markdown(
        f"**[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)** (distractor "
        "config), the same dataset and the same 40-question stratified sample as "
        "Agentic Architectures, reused deliberately: every question needs facts from two "
        "different paragraphs to answer, and every question ships 10 candidate paragraphs, "
        "2 gold and 8 distractors, forcing a real test of whether a shape can find and use "
        "more than one fact. The `search()` tool returns only its single best-matching "
        "paragraph (k=1) and is identical across every shape here, so any difference in "
        "results comes from what each shape can *decide to do* with that tool, not from a "
        "different or better retriever.\n\n"
        f"Every shape answers the same **{records['question_id'].nunique()} sampled "
        "questions**, split evenly between **bridge** (hop 2 needs an entity hop 1 finds) "
        "and **comparison** (two independent lookups compared against each other)."
    )

    st.header("Shapes compared")
    for meta in shapes_meta.values():
        st.markdown(f"**{meta['name']}:** {meta['description']}")
    st.caption(
        "One representative technique per shape, not every technique from the prior two "
        "experiments. RAG's representative is single-shot retrieval (the simplest, "
        "non-adaptive form); the multi-agent representative is Supervisor + Verification "
        "Loop, reused because it won the agentic-architectures comparison outright, the "
        "strongest real multi-agent contender available, not a strawman."
    )

    st.header("Metrics, and the one this experiment adds")
    st.markdown(
        "The same three categories as this repo's other experiments (quality, agentic, "
        "operational), plus **redundant step rate**: a later tool call retrieving a "
        "paragraph title an earlier tool call in the same run already retrieved. This tests "
        "whether [MAST](https://arxiv.org/abs/2503.13657)'s most common real-world "
        "multi-agent failure mode, *step repetition* (17.14% of failures in that paper's "
        "study), shows up in this repo's own architectures. It's a narrower, single-tool "
        "operationalization of \"redoing work already done\", not every way a system can "
        "repeat itself; see [EVALUATION_CRITERIA.md](../../EVALUATION_CRITERIA.md) for the "
        "gap this metric was built to close."
    )
    st.markdown(
        "- **Quality**: Exact Match and F1 against the gold answer, split by question type\n"
        "- **Agentic**: step/tool-call count, tool error rate, handoff count, early "
        "termination rate, redundant step rate, state overhead\n"
        "- **Operational**: tokens, latency, time to first token, context payload, empty "
        "(non-gold) retrieval rate, model-call error rate"
    )
    st.caption(
        "Semantic cache hit rate isn't tracked: nothing in this experiment caches across "
        "questions, so the metric wouldn't mean anything here, the same reasoning as the "
        "prior two experiments."
    )

    st.header("Model backend")
    st.markdown(
        "Every model call in every shape goes through the same local "
        "[Ollama](https://ollama.com) server running `llama3.1:8b` (see `llm_client.py`), "
        "so differences in the results come from shape, not from which model answered."
    )

with tab_comparison:
    quality = quality_summary(records)
    agentic = agentic_summary(records)
    operational = operational_summary(records)

    st.header("Quick reference: strongest and weakest metric")
    st.caption(
        "For each shape, its top 2 metrics that actually distinguish it from the other three "
        "(Worked well) and its bottom 2 (Struggled). A metric every shape is tied on is "
        "excluded: it isn't a distinguishing strength or weakness for any one of them."
    )

    def describe_side(key: str, side: str) -> tuple[str, str]:
        picks = notable_metrics(combined, key, side)
        labels = [f"{METRIC_LABELS[m]}: {format_metric(m, v)} (rank {r}/{n})" for m, v, r, n in picks]
        while len(labels) < 2:
            labels.append("Nothing standout in this sample")
        return labels[0], labels[1]

    overview_rows = []
    for key, meta in shapes_meta.items():
        best_1, best_2 = describe_side(key, "best")
        worst_1, worst_2 = describe_side(key, "worst")
        overview_rows.append(
            {"Shape": meta["name"], "Worked well": best_1, "Also strong": best_2, "Struggled": worst_1, "Also weak": worst_2}
        )
    st.dataframe(pd.DataFrame(overview_rows).set_index("Shape"), width="stretch")

    st.header("Supporting data")

    st.subheader("Quality: all shapes")
    st.dataframe(quality.rename(columns=METRIC_LABELS).style.format("{:.1%}"), width="stretch")
    st.bar_chart(quality[["exact_match", "f1", "bridge_f1", "comparison_f1"]])

    st.subheader("Agentic cost: all shapes")
    agentic_formats = {METRIC_LABELS[c]: METRIC_FORMATS.get(c, "{:.2f}") for c in agentic.columns}
    st.dataframe(agentic.rename(columns=METRIC_LABELS).style.format(agentic_formats), width="stretch")
    st.bar_chart(agentic[["mean_step_count", "mean_handoffs"]])
    st.bar_chart(agentic[["early_termination_rate", "redundant_step_rate", "tool_error_rate"]])

    st.subheader("Operational cost: all shapes")
    operational_formats = {METRIC_LABELS[c]: METRIC_FORMATS.get(c, "{:.2f}") for c in operational.columns}
    st.dataframe(operational.rename(columns=METRIC_LABELS).style.format(operational_formats), width="stretch")
    st.bar_chart(operational[["mean_latency_seconds"]])
    st.bar_chart(operational[["mean_prompt_tokens", "mean_completion_tokens"]])

    st.subheader("Accuracy vs. wall-clock time")
    tradeoff = quality[["exact_match"]].join(agentic[["mean_wall_clock_seconds"]])
    st.scatter_chart(tradeoff, x="mean_wall_clock_seconds", y="exact_match")

    st.caption(
        "40 questions is enough to see clear directional differences, not enough for tight "
        "statistical confidence on the exact percentage-point gaps between shapes."
    )

with tab_deep_dive:
    shape_key = st.selectbox(
        "Shape", options=list(SHAPES.keys()), format_func=lambda k: shapes_meta[k]["name"]
    )
    meta = shapes_meta[shape_key]
    st.subheader(meta["name"])

    st.markdown("#### What it is")
    st.markdown(meta["what_it_is"])

    diagram_heights = {
        "plain_llm": 120,
        "single_shot_rag": 140,
        "single_agent_react": 280,
        "multi_agent_supervisor": 200,
    }
    render_mermaid(meta["diagram"], height=diagram_heights[shape_key])

    st.markdown("#### How we implemented it")
    st.markdown(meta["how_we_implemented_it"])

    st.markdown("#### When it's useful")
    st.markdown(meta["when_its_useful"])

    st.markdown("#### How it ranks against the other three shapes, best to worst")
    ranked = ranked_metrics(combined, shape_key)
    ranked_table = pd.DataFrame(
        [{"Metric": METRIC_LABELS[m], "Value": format_metric(m, v), "Rank": f"{r} of {n}"} for m, v, r, n in ranked]
    )
    st.dataframe(ranked_table.set_index("Metric"), width="stretch")

    st.markdown("#### Prompts used, verbatim")
    for role, prompt in meta["prompts"].items():
        with st.expander(f"{role}"):
            st.text(prompt)

    st.markdown("#### Example runs")
    st.caption("Browse this shape's actual traces on real questions from the sample.")
    shape_records = records[records["shape"] == shape_key]
    picked_question = st.selectbox(
        "Question", options=shape_records["question_id"].tolist(),
        format_func=lambda qid: shape_records.set_index("question_id").loc[qid, "question"][:80],
        key=f"deep_dive_question_{shape_key}",
    )
    example_run = shape_records[shape_records["question_id"] == picked_question].iloc[0]
    st.markdown(f"**Gold answer:** `{example_run['gold_answer']}` · **Predicted:** `{format_predicted(example_run['predicted_answer'])}` · **F1:** {example_run['f1']:.2f}")
    render_trace(example_run["steps"])

with tab_explorer:
    st.header("Compare every shape on one question")
    st.caption("Pick a question and see how all four shapes answered it, side by side.")

    question_ids = records["question_id"].unique().tolist()
    question_labels = records.drop_duplicates("question_id").set_index("question_id")["question"]
    picked = st.selectbox("Question", options=question_ids, format_func=lambda qid: question_labels[qid][:100])

    question_records = records[records["question_id"] == picked].set_index("shape")
    gold_answer = question_records.iloc[0]["gold_answer"]
    question_type = question_records.iloc[0]["question_type"]

    st.markdown(f"**Question:** {question_labels[picked]}")
    st.markdown(f"**Gold answer:** `{gold_answer}` · **Type:** `{question_type}`")

    display_summary = pd.DataFrame(
        {
            "Predicted": question_records["predicted_answer"].map(format_predicted),
            "Correct": question_records["exact_match"].map({1.0: "✅", 0.0: "❌"}),
            "F1": question_records["f1"].round(2),
            "Tool calls": question_records["tool_calls"],
            "Redundant retrievals": question_records["redundant_retrievals"],
        }
    ).rename(index=lambda k: shapes_meta[k]["name"])
    st.dataframe(display_summary, width="stretch")

    for shape_key in SHAPES:
        with st.expander(f"{shapes_meta[shape_key]['name']}, full trace"):
            render_trace(question_records.loc[shape_key, "steps"])
