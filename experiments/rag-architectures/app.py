"""Streamlit dashboard: methodology, cross-architecture comparison (overall
and broken down by MultiHop-RAG's 4 question types), an architecture deep
dive (verbatim prompts + full step-by-step traces), a query explorer for
comparing every architecture's answer to the same query side by side, the
chunking sub-experiment, and a written embedding-quality guide.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from metrics import operational_summary, quality_by_question_type, quality_summary, retrieval_summary
from techniques import ARCHITECTURES

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"
CHUNKING_RESULTS_PATH = Path(__file__).parent / "results" / "chunking_records.json"

QUESTION_TYPE_LABELS = {
    "inference_query": "Inference",
    "comparison_query": "Comparison",
    "temporal_query": "Temporal",
    "null_query": "Null (unanswerable)",
}

METRIC_LABELS = {
    "exact_match": "Exact Match",
    "f1": "F1",
    "incorrect_abstention_rate": "Incorrect Abstention Rate (answerable queries)",
    "correct_abstention_rate": "Correct Abstention Rate (null queries)",
    "faithfulness": "Faithfulness (lexical overlap proxy)",
    "recall_at_k": "Recall@k",
    "precision_at_k": "Precision@k",
    "mrr": "MRR",
    "mean_llm_calls": "Mean LLM Calls",
    "mean_retrieval_rounds": "Mean Retrieval Rounds",
    "mean_prompt_tokens": "Mean Prompt Tokens",
    "mean_completion_tokens": "Mean Completion Tokens",
    "mean_latency_seconds": "Mean Model Latency (s)",
    "median_time_to_first_token_seconds": "Median Time to First Token (s)",
    "mean_context_payload_bytes": "Mean Context Payload (bytes)",
    "mean_wall_clock_seconds": "Mean Wall-Clock Time (s)",
    "empty_retrieval_rate": "Empty Retrieval Rate",
    "error_rate": "Model Call Error Rate",
}
METRIC_FORMATS = {
    "exact_match": "{:.1%}", "f1": "{:.1%}", "incorrect_abstention_rate": "{:.1%}",
    "correct_abstention_rate": "{:.1%}", "faithfulness": "{:.1%}",
    "recall_at_k": "{:.1%}", "precision_at_k": "{:.1%}", "mrr": "{:.2f}",
    "mean_llm_calls": "{:.1f}", "mean_retrieval_rounds": "{:.2f}",
    "mean_prompt_tokens": "{:.0f}", "mean_completion_tokens": "{:.0f}",
    "mean_latency_seconds": "{:.1f}", "median_time_to_first_token_seconds": "{:.2f}",
    "mean_context_payload_bytes": "{:.0f}", "mean_wall_clock_seconds": "{:.1f}",
    "empty_retrieval_rate": "{:.1%}", "error_rate": "{:.1%}",
}
LOWER_IS_BETTER = {
    "incorrect_abstention_rate", "mean_llm_calls", "mean_retrieval_rounds",
    "mean_prompt_tokens", "mean_completion_tokens", "mean_latency_seconds",
    "median_time_to_first_token_seconds", "mean_context_payload_bytes",
    "mean_wall_clock_seconds", "empty_retrieval_rate", "error_rate",
}

st.set_page_config(page_title="RAG Architectures Compared", layout="wide")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame, str, int]:
    payload = json.loads(RESULTS_PATH.read_text())
    return payload["architectures"], pd.DataFrame(payload["runs"]), payload["chunking_strategy"], payload["chunk_count"]


@st.cache_data
def load_chunking_results() -> tuple[dict, pd.DataFrame]:
    payload = json.loads(CHUNKING_RESULTS_PATH.read_text())
    return payload["strategies"], pd.DataFrame(payload["runs"])


def combined_metrics(records: pd.DataFrame) -> pd.DataFrame:
    return quality_summary(records).join(retrieval_summary(records)).join(operational_summary(records))


def format_metric(metric: str, value) -> str:
    if pd.isna(value):
        return "–"
    return METRIC_FORMATS.get(metric, "{:.2f}").format(value)


def format_predicted(value) -> str:
    return "(no answer)" if pd.isna(value) else str(value)


def ranked_metrics(combined: pd.DataFrame, architecture_key: str) -> list[tuple[str, float, int, int]]:
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


def notable_metrics(combined: pd.DataFrame, architecture_key: str, side: str, count: int = 3) -> list[tuple[str, float, int, int]]:
    picks = []
    for metric, value, rank, count_n in ranked_metrics(combined, architecture_key):
        column = combined[metric].dropna()
        if column.max() == column.min():
            continue
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


def render_header() -> None:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #5B21B6, #DB2777); padding:1.5rem 1.5rem;
                    border-radius:0.5rem; margin-bottom:1.5rem;">
            <h1 style="color:white; margin:0;">RAG Architectures Compared</h1>
            <p style="color:white; margin:0.25rem 0 0 0; opacity:0.9;">
                6 retrieval-augmented generation designs, one shared corpus, one shared question set.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prompt_block(label: str, prompt: str | None) -> None:
    if prompt is None:
        return
    with st.expander(f"Prompt: {label}", expanded=False):
        st.code(prompt, language=None)


def render_step_trace(steps: list[dict]) -> None:
    for i, step in enumerate(steps, start=1):
        icon = "🔎" if step["kind"] == "retrieval_call" else "💬"
        header = f"{icon} Step {i}: {step['role']} ({step['kind']})"
        with st.expander(header, expanded=False):
            st.markdown(f"**Detail:** {step['detail']}")
            if step["kind"] == "retrieval_call":
                st.markdown(f"**Chunks retrieved:** {len(step['retrieved_chunk_ids'])}")
                if step["output"]:
                    st.markdown(f"**Titles:** {step['output']}")
            else:
                st.markdown(f"**Output:** {step['output']}")
                cols = st.columns(4)
                cols[0].metric("Prompt tokens", step["prompt_tokens"])
                cols[1].metric("Completion tokens", step["completion_tokens"])
                cols[2].metric("Latency (s)", f"{step['latency_seconds']:.2f}")
                cols[3].metric("Context bytes", step["context_payload_bytes"])
            if step["error"]:
                st.error(f"Error: {step['error']}")


def main() -> None:
    render_header()

    if not RESULTS_PATH.exists():
        st.warning("No results yet. Run `python3 run_experiment.py` first.")
        return

    architectures_meta, records, chunking_strategy, chunk_count = load_results()
    combined = combined_metrics(records)
    by_type = quality_by_question_type(records)

    tabs = st.tabs(["Methodology", "Comparison", "Architecture Deep Dive", "Query Explorer", "Chunking Experiment", "Embedding Quality"])

    # ---- Methodology ---------------------------------------------------
    with tabs[0]:
        st.header("Methodology")
        st.markdown(
            f"""
**Dataset:** [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) — 609 news
articles as the corpus, {len(records['question_id'].unique())} queries sampled evenly across its
4 labeled question types: **Inference**, **Comparison**, **Temporal**, and **Null** (deliberately
unanswerable from the corpus, to test whether an architecture hallucinates or abstains).

**Corpus indexing:** every architecture shares one index, built once with the **{chunking_strategy}**
chunking strategy ({chunk_count} chunks total). That's the one thing every architecture has in
common; the chunking sub-experiment (see its own tab) is what varies this instead of holding it fixed.

**Model:** `llama3.1:8b` (Q4_K_M) via [Ollama](https://ollama.com), embeddings via
`all-MiniLM-L6-v2`, reranking via `cross-encoder/ms-marco-MiniLM-L-6-v2` — all local, all free.

**Answer prompt:** every architecture uses the exact same final-answer prompt (see each
architecture's Deep Dive tab), told to answer only from the passages it's given and to say
"Insufficient information." when they don't support an answer. Any difference in how often an
architecture abstains comes from what it retrieved and how, not from different instructions.
            """
        )
        st.subheader("Architectures")
        for key, module in ARCHITECTURES.items():
            st.markdown(f"**{module.NAME}** — {module.DESCRIPTION}")

    # ---- Comparison ------------------------------------------------------
    with tabs[1]:
        st.header("Cross-Architecture Comparison")

        st.subheader("Overall metrics")
        display = combined.copy()
        for metric in display.columns:
            display[metric] = display[metric].map(lambda v, m=metric: format_metric(m, v))
        display = display.rename(columns=METRIC_LABELS)
        st.dataframe(display, width="stretch")

        st.subheader("Which architecture wins at which scenario")
        st.markdown(
            "F1 for the 3 answerable question types, and abstention rate for null queries "
            "(higher is better in every column here, including abstention rate on null queries)."
        )
        by_type_display = by_type.copy()
        for col in by_type_display.columns:
            metric_key = "f1" if col.endswith("_f1") else "correct_abstention_rate"
            by_type_display[col] = by_type_display[col].map(lambda v, m=metric_key: format_metric(m, v))
        by_type_display.columns = [
            QUESTION_TYPE_LABELS.get(c.replace("_f1", ""), "Null Query Abstention Rate") for c in by_type.columns
        ]
        st.dataframe(by_type_display, width="stretch")

        st.subheader("Best / worst by architecture")
        for architecture_key in combined.index:
            name = architectures_meta[architecture_key]["name"]
            best = notable_metrics(combined, architecture_key, "best")
            worst = notable_metrics(combined, architecture_key, "worst")
            with st.expander(name, expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Strongest metrics**")
                    for metric, value, rank, n in best:
                        st.markdown(f"- {METRIC_LABELS.get(metric, metric)}: {format_metric(metric, value)} (rank {rank}/{n})")
                with col2:
                    st.markdown("**Weakest metrics**")
                    for metric, value, rank, n in worst:
                        st.markdown(f"- {METRIC_LABELS.get(metric, metric)}: {format_metric(metric, value)} (rank {rank}/{n})")

    # ---- Architecture Deep Dive -------------------------------------------
    with tabs[2]:
        st.header("Architecture Deep Dive")
        architecture_key = st.selectbox(
            "Architecture", options=list(architectures_meta.keys()),
            format_func=lambda k: architectures_meta[k]["name"],
        )
        meta = architectures_meta[architecture_key]
        st.subheader(meta["name"])
        st.markdown(meta["description"])
        render_mermaid(meta["diagram"])

        st.markdown("### What it is")
        st.markdown(meta["what_it_is"])
        st.markdown("### How we implemented it")
        st.markdown(meta["how_we_implemented_it"])
        st.markdown("### When it's useful")
        st.markdown(meta["when_its_useful"])

        st.markdown("### Prompts used (verbatim)")
        for role, prompt in meta["prompts"].items():
            render_prompt_block(role, prompt)

        st.markdown("### Metrics")
        row = combined.loc[[architecture_key]].copy()
        for metric in row.columns:
            row[metric] = row[metric].map(lambda v, m=metric: format_metric(m, v))
        st.dataframe(row.rename(columns=METRIC_LABELS), width="stretch")

    # ---- Query Explorer ----------------------------------------------
    with tabs[3]:
        st.header("Query Explorer")
        st.markdown("See every architecture's answer to the same query, side by side.")

        question_ids = records["question_id"].unique().tolist()
        question_lookup = records.drop_duplicates("question_id").set_index("question_id")

        type_filter = st.multiselect(
            "Filter by question type",
            options=list(QUESTION_TYPE_LABELS.keys()),
            default=list(QUESTION_TYPE_LABELS.keys()),
            format_func=lambda k: QUESTION_TYPE_LABELS[k],
        )
        filtered_ids = [qid for qid in question_ids if question_lookup.loc[qid, "question_type"] in type_filter]

        question_id = st.selectbox(
            "Query", options=filtered_ids,
            format_func=lambda qid: f"[{QUESTION_TYPE_LABELS[question_lookup.loc[qid, 'question_type']]}] {question_lookup.loc[qid, 'question'][:100]}",
        )
        row0 = question_lookup.loc[question_id]
        st.markdown(f"**Question:** {row0['question']}")
        st.markdown(f"**Gold answer:** {row0['gold_answer']}")
        st.markdown(f"**Type:** {QUESTION_TYPE_LABELS[row0['question_type']]}")

        query_records = records[records["question_id"] == question_id]
        for _, rec in query_records.iterrows():
            with st.expander(f"{rec['architecture_name']}: {format_predicted(rec['predicted_answer'])}", expanded=False):
                cols = st.columns(4)
                cols[0].metric("F1", format_metric("f1", rec["f1"]) if pd.notna(rec["f1"]) else "n/a")
                cols[1].metric("Recall@k", format_metric("recall_at_k", rec["recall_at_k"]) if pd.notna(rec["recall_at_k"]) else "n/a")
                cols[2].metric("LLM calls", rec["llm_calls"])
                cols[3].metric("Wall-clock (s)", f"{rec['wall_clock_seconds']:.1f}")
                render_step_trace(rec["steps"])

    # ---- Chunking Experiment -------------------------------------------
    with tabs[4]:
        st.header("Chunking Sub-Experiment")
        if not CHUNKING_RESULTS_PATH.exists():
            st.warning("No chunking results yet. Run `python3 run_chunking_experiment.py` first.")
        else:
            strategies_meta, chunking_records = load_chunking_results()
            st.markdown(
                "Architecture held fixed at **Naive RAG**; only the chunking strategy changes. "
                "This isolates what chunking alone does to retrieval and answer quality, without "
                "a reranker or fusion step able to compensate for a weak first-pass chunk."
            )

            st.subheader("Chunk statistics by strategy")
            stats_df = pd.DataFrame(strategies_meta).T
            stats_df.columns = ["Chunk Count", "Mean Tokens/Chunk", "Min Tokens/Chunk", "Max Tokens/Chunk"]
            st.dataframe(stats_df, width="stretch")

            st.subheader("Retrieval and answer quality by strategy")
            rows = []
            for strategy, group in chunking_records.groupby("strategy"):
                answerable = group[group["question_type"] != "null_query"]
                null_group = group[group["question_type"] == "null_query"]
                rows.append(
                    {
                        "strategy": strategy,
                        "recall_at_k": answerable["recall_at_k"].mean(),
                        "precision_at_k": answerable["precision_at_k"].mean(),
                        "mrr": answerable["mrr"].mean(),
                        "f1": answerable["f1"].mean(),
                        "faithfulness": answerable["faithfulness"].mean(),
                        "correct_abstention_rate": null_group["abstained"].mean() if len(null_group) else None,
                    }
                )
            chunk_summary = pd.DataFrame(rows).set_index("strategy")
            chunk_summary_display = chunk_summary.copy()
            for metric in chunk_summary_display.columns:
                chunk_summary_display[metric] = chunk_summary_display[metric].map(lambda v, m=metric: format_metric(m, v))
            st.dataframe(chunk_summary_display.rename(columns=METRIC_LABELS), width="stretch")
            st.bar_chart(chunk_summary[["recall_at_k", "f1"]])

    # ---- Embedding Quality -----------------------------------------------
    with tabs[5]:
        st.header("Ensuring Embedding Quality")
        st.markdown(
            """
Every architecture in this experiment sits on top of one embedding model
(`all-MiniLM-L6-v2`), and every one of them fails the same way if that
embedding is weak: the wrong chunks get retrieved, and no amount of
reranking, fusion, or clever prompting downstream can recover information
that was never retrieved in the first place. A few practical checks:

**Pick a model benchmarked on retrieval, not just general similarity.**
The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) reports a
dedicated Retrieval score, separate from its Classification or STS scores. A
model that's strong at general sentence similarity isn't necessarily strong
at the specific job of ranking a passage against a query, which is a
distinct, asymmetric task.

**Match query/passage asymmetry if your model supports it.** A query
("who criticized the merger") and the passage that answers it ("Jane Doe
called the merger reckless") are different kinds of text. Some embedding
models (e.g. the `e5` and `bge` families) are trained with separate
`query:` and `passage:` prefixes specifically so the model can treat the two
differently; using a symmetric model, or forgetting the prefix on an
asymmetric one, throws that signal away.

**Validate retrieval quality directly, don't assume it from the model
card.** The only real test is Recall@k and MRR on a labeled sample of your
own corpus and queries, exactly what `retrieval_summary()` computes in this
experiment. A benchmark score measures the model in general; it doesn't
guarantee the model understands your corpus's specific vocabulary
(product names, internal jargon, a news corpus's proper nouns).

**Chunk size interacts with embedding quality, not just retrieval
scope.** An embedding is one fixed-size vector standing in for everything
in the chunk. A chunk that's too long packs multiple topics into that one
vector, diluting it until it doesn't strongly match any single query. A
chunk that's too short can lose the surrounding context that gives a
sentence its actual meaning. See the Chunking Experiment tab: this is the
same effect that sub-experiment measures directly.

**Re-validate after any embedding model change.** Swapping model versions
changes the vector space; old embeddings and new embeddings are not
comparable. A corpus embedded with model version A must be **fully
re-embedded**, not incrementally updated, before being queried with model
version B, or retrieval quality degrades silently, with no error thrown.

**Normalize consistently.** This experiment L2-normalizes every embedding
(`normalize_embeddings=True`) so a plain dot product is equivalent to
cosine similarity. Mixing normalized and unnormalized vectors in the same
index silently corrupts every similarity score.
            """
        )


if __name__ == "__main__":
    main()
