"""Streamlit dashboard: methodology, cross-technique comparison (overall,
by document length, by needle position, and the full length x position
grid), a technique deep dive (verbatim prompts + how each technique
works), and a document explorer for comparing every technique's answer
to the same document side by side.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from metrics import operational_summary, quality_by_length, quality_by_length_and_position, quality_by_position, quality_summary
from techniques import TECHNIQUES

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"

LENGTH_ORDER = ["short", "medium", "long"]
POSITION_ORDER = ["start", "middle", "end"]
LENGTH_LABELS = {"short": "Short (~500 words)", "medium": "Medium (~2,000 words)", "long": "Long (~6,000 words)"}
POSITION_LABELS = {"start": "Start", "middle": "Middle", "end": "End"}

METRIC_LABELS = {
    "exact_match": "Exact Match",
    "f1": "F1",
    "mean_llm_calls": "Mean LLM Calls",
    "mean_prompt_tokens": "Mean Prompt Tokens",
    "mean_completion_tokens": "Mean Completion Tokens",
    "mean_latency_seconds": "Mean Model Latency (s)",
    "median_time_to_first_token_seconds": "Median Time to First Token (s)",
    "mean_context_payload_bytes": "Mean Context Payload (bytes)",
    "mean_wall_clock_seconds": "Mean Wall-Clock Time (s)",
    "error_rate": "Model Call Error Rate",
}
METRIC_FORMATS = {
    "exact_match": "{:.1%}", "f1": "{:.1%}",
    "mean_llm_calls": "{:.1f}", "mean_prompt_tokens": "{:.0f}", "mean_completion_tokens": "{:.0f}",
    "mean_latency_seconds": "{:.1f}", "median_time_to_first_token_seconds": "{:.2f}",
    "mean_context_payload_bytes": "{:.0f}", "mean_wall_clock_seconds": "{:.1f}", "error_rate": "{:.1%}",
}
LOWER_IS_BETTER = {
    "mean_llm_calls", "mean_prompt_tokens", "mean_completion_tokens", "mean_latency_seconds",
    "median_time_to_first_token_seconds", "mean_context_payload_bytes", "mean_wall_clock_seconds", "error_rate",
}

CONCEPT_TERMINOLOGY = {
    "Context window": (
        "The total number of tokens (roughly, word-pieces) a model can attend to in one request, "
        "input and output combined. Every call is stateless: whatever isn't inside the window "
        "simply doesn't exist to the model on that call, no matter how important it was earlier."
    ),
    "Needle-in-a-haystack test": (
        "A synthetic way to test long-context handling: bury one small, specific piece of "
        "information (the \"needle\", here a short SQuAD paragraph the question is actually about) "
        "inside a much larger pile of unrelated filler text (the \"haystack\"), then ask a question "
        "that can only be answered using the needle. Varying the haystack's size and where the "
        "needle sits inside it isolates *length* and *position* as independent, controllable "
        "variables, which a naturally-long real document can't offer."
    ),
    "Lost in the middle": (
        "A documented pattern ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)) where a "
        "model's accuracy at using a piece of information is highest when that information sits "
        "near the start or end of its context, and measurably worse when the same information sits "
        "in the middle, even though every token is technically inside the window the whole time."
    ),
    "Context management technique": (
        "Any strategy for deciding what actually goes into a model's limited context window when "
        "the available content is larger than what's worth (or possible) to send in full: keep "
        "everything, keep only what's recent, keep only what's relevant, or compress what's old "
        "into a shorter summary. This experiment compares 4 of them head to head."
    ),
}

TECHNIQUE_METRIC_TERMINOLOGY = {
    "Exact Match / F1": (
        "Two ways of scoring a predicted answer against the correct one. Exact Match is strict: 1 "
        "if the (normalized) strings match exactly, 0 otherwise. F1 gives partial credit based on "
        "word overlap, so an answer that's mostly right but adds or drops a word still scores above "
        "zero."
    ),
    "LLM calls": (
        "How many separate calls to the language model one document needed. Full Context, Sliding "
        "Window, and Retrieval Selection always make exactly 1. Hierarchical Summarization makes "
        "one extra call per chunk boundary it has to fold into its running summary, so its call "
        "count grows with document length while the other three stay flat."
    ),
    "Prompt tokens / completion tokens": (
        "How much text, measured in tokens (roughly, word-pieces), a call sends to the model "
        "(prompt) and gets back (completion). This is the direct cost signal: a technique that "
        "sends fewer prompt tokens per document is cheaper to run, independent of whether it "
        "answers correctly."
    ),
    "Latency / wall-clock time": (
        "Latency is how long a single model call took, start to finish. Wall-clock time is how "
        "long the *whole* document took, including every call a technique made (more than one, for "
        "Hierarchical Summarization), which is what actually matters to whoever is waiting on an "
        "answer."
    ),
    "Time to first token (TTFT)": (
        "How long a model takes to start responding, as opposed to latency, which includes the "
        "time to finish the whole response. This dashboard reports the *median* TTFT rather than "
        "the mean, since Ollama's self-reported prefill timing can spike under local memory "
        "pressure; a few spikes would swamp a mean but not a median. See the README for how KV "
        "cache reuse also skews this number specifically for local serving."
    ),
    "Context payload bytes": (
        "The size, in bytes, of everything sent to the model in one call: the system prompt, the "
        "text, the question. A proxy for how much the model has to read before it can answer, "
        "measured before tokenization."
    ),
    "Error rate": (
        "How often a model call failed outright (a timeout, a malformed response, a dropped "
        "connection), as opposed to succeeding but giving a wrong answer."
    ),
}

st.set_page_config(page_title="Context Window Management Compared", layout="wide")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame, dict]:
    payload = json.loads(RESULTS_PATH.read_text())
    return payload["techniques"], pd.DataFrame(payload["runs"]), {"needle_count": payload["needle_count"], "lengths": payload["lengths"]}


def combined_metrics(records: pd.DataFrame) -> pd.DataFrame:
    return quality_summary(records).join(operational_summary(records))


def format_metric(metric: str, value) -> str:
    if pd.isna(value):
        return "–"
    return METRIC_FORMATS.get(metric, "{:.2f}").format(value)


def reindex_lengths(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=[c for c in LENGTH_ORDER if c in df.columns]).rename(columns=LENGTH_LABELS)


def reindex_positions(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=[c for c in POSITION_ORDER if c in df.columns]).rename(columns=POSITION_LABELS)


def render_mermaid(diagram: str, height: int = 220) -> None:
    # See rag-architectures/app.py's render_mermaid for the full writeup of
    # the three real rendering bugs this works around (auto-init timing,
    # requestAnimationFrame pausing in hidden tabs, and a premature
    # degeneracy check racing mermaid.run()'s own layout pass). Kept
    # byte-for-byte identical here on purpose, this is the site's shared
    # diagram-rendering fix, not a per-experiment one.
    components.html(
        f"""
        <div id="diagram-container" style="font-family: sans-serif;"></div>
        <script id="diagram-source" type="text/plain">{diagram}</script>
        <script>
            var diagramSource = document.getElementById("diagram-source").textContent;

            function isDegenerate(container) {{
                var svg = container.querySelector("svg");
                if (!svg) return true;
                var vb = svg.viewBox && svg.viewBox.baseVal;
                return !vb || vb.width < 50 || vb.height < 20;
            }}

            function renderOnce(container) {{
                container.removeAttribute("data-processed");
                container.className = "mermaid";
                container.textContent = diagramSource;
                mermaid.initialize({{ startOnLoad: false, flowchart: {{ useMaxWidth: true }} }});
                mermaid.run({{ nodes: [container] }});
            }}

            function doRender() {{
                var container = document.getElementById("diagram-container");
                renderOnce(container);
                setTimeout(function () {{
                    if (!isDegenerate(container)) return;
                    renderOnce(container);
                    setTimeout(function () {{
                        if (isDegenerate(container)) {{
                            container.innerText =
                                "Diagram didn't render correctly. Reloading the page usually fixes this.";
                        }}
                    }}, 1200);
                }}, 1200);
            }}

            function waitForStableWidth(startedAt, lastWidth, stableCount, attemptsLeft) {{
                var width = document.getElementById("diagram-container").offsetWidth;
                var stable = width > 0 && width === lastWidth;
                var minTimeElapsed = (Date.now() - startedAt) >= 1500;
                if ((stable && stableCount >= 5 && minTimeElapsed) || attemptsLeft <= 0) {{
                    document.fonts.ready.then(doRender);
                }} else {{
                    setTimeout(function () {{
                        waitForStableWidth(startedAt, width, stable ? stableCount + 1 : 0, attemptsLeft - 1);
                    }}, 16);
                }}
            }}

            function loadMermaid(retriesLeft) {{
                var script = document.createElement("script");
                script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
                script.onload = function () {{ waitForStableWidth(Date.now(), -1, 0, 400); }};
                script.onerror = function () {{
                    if (retriesLeft > 0) {{
                        setTimeout(function () {{ loadMermaid(retriesLeft - 1); }}, 500);
                    }} else {{
                        document.getElementById("diagram-container").innerText =
                            "Diagram failed to load (network issue reaching the mermaid CDN). Reloading the page usually fixes this.";
                    }}
                }};
                document.head.appendChild(script);
            }}
            loadMermaid(4);
        </script>
        """,
        height=height,
    )


def render_header() -> None:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #5B21B6, #DB2777); padding:1.5rem 1.5rem;
                    border-radius:0.5rem; margin-bottom:1.5rem;">
            <h1 style="color:white; margin:0;">Context Window Management Compared</h1>
            <p style="color:white; margin:0.25rem 0 0 0; opacity:0.9;">
                4 ways to decide what stays in context, tested against a controlled needle-in-a-haystack grid.
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
        icon = "📝" if step["role"] == "summarizer" else "💬"
        with st.expander(f"{icon} Step {i}: {step['role']}", expanded=False):
            st.markdown(f"**Detail:** {step['detail']}")
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

    techniques_meta, records, run_info = load_results()
    combined = combined_metrics(records)
    by_length = quality_by_length(records)
    by_position = quality_by_position(records)
    grid = quality_by_length_and_position(records)

    tabs = st.tabs(["Methodology", "Terminology", "Comparison", "Technique Deep Dive", "Document Explorer"])

    # ---- Methodology -----------------------------------------------------
    with tabs[0]:
        st.header("Methodology")
        st.markdown(
            f"""
**Dataset:** [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) — {run_info['needle_count']} short
paragraph+question+answer triples sampled as "needles", buried inside a pile of other, unrelated
SQuAD paragraphs (the "haystack") built purely for this experiment. Every needle is tested at 3
document lengths ({", ".join(f"{v} words ({k})" for k, v in run_info['lengths'].items())}) and 3
needle positions (start, middle, end), for a full length x position grid, run through all 4
techniques below.

**Model:** `llama3.1:8b` (Q4_K_M) via [Ollama](https://ollama.com), embeddings for Retrieval
Selection via `all-MiniLM-L6-v2`, both local and free.

**Answer prompt:** every technique uses the exact same final-answer prompt (see each technique's
Deep Dive tab), told to answer only from the text it's given. Any difference in accuracy comes
from what each technique put in front of the model, not from different instructions.
            """
        )
        st.subheader("Techniques")
        for key, module in TECHNIQUES.items():
            st.markdown(f"**{module.NAME}** — {module.DESCRIPTION}")

    # ---- Terminology -------------------------------------------------------
    with tabs[1]:
        st.header("Terminology")
        st.caption("Definitions for the context-management concepts used throughout this dashboard. Metric definitions live in the Comparison tab, next to the tables they describe.")
        for term, definition in CONCEPT_TERMINOLOGY.items():
            with st.expander(term):
                st.markdown(definition)

    # ---- Comparison --------------------------------------------------------
    with tabs[2]:
        st.header("Cross-Technique Comparison")

        st.subheader("Overall metrics")
        display = combined.copy()
        for metric in display.columns:
            display[metric] = display[metric].map(lambda v, m=metric: format_metric(m, v))
        st.dataframe(display.rename(columns=METRIC_LABELS), width="stretch")

        st.subheader("Accuracy by document length")
        st.caption("Mean F1, averaged across all 3 needle positions. Isolates the pure length effect.")
        st.dataframe(reindex_lengths(by_length).map(lambda v: format_metric("f1", v)), width="stretch")

        st.subheader("Accuracy by needle position")
        st.caption("Mean F1, averaged across all 3 document lengths. This is the table that shows (or doesn't show) lost in the middle, for each technique.")
        st.dataframe(reindex_positions(by_position).map(lambda v: format_metric("f1", v)), width="stretch")

        st.subheader("Full length x position grid")
        st.caption("Every (length, position) cell, one row per technique — the complete picture neither breakdown above shows alone.")
        technique_choice = st.selectbox(
            "Technique", options=list(techniques_meta.keys()), format_func=lambda k: techniques_meta[k]["name"], key="grid_technique"
        )
        pivot = grid[grid["technique"] == technique_choice].pivot(index="position", columns="length", values="f1")
        pivot = pivot.reindex(index=[p for p in POSITION_ORDER if p in pivot.index], columns=[l for l in LENGTH_ORDER if l in pivot.columns])
        st.dataframe(pivot.rename(columns=LENGTH_LABELS, index=POSITION_LABELS).map(lambda v: format_metric("f1", v)), width="stretch")

    # ---- Technique Deep Dive ------------------------------------------------
    with tabs[3]:
        st.header("Technique Deep Dive")
        st.caption("All 4 techniques, side by side — expand any one to see how it works, its prompts, and its own metrics.")
        for key, meta in techniques_meta.items():
            with st.expander(meta["name"], expanded=False):
                st.markdown(meta["description"])
                render_mermaid(meta["diagram"], height=200)

                st.markdown("**What it is**")
                st.markdown(meta["what_it_is"])
                st.markdown("**How we implemented it**")
                st.markdown(meta["how_we_implemented_it"])
                st.markdown("**When it's useful**")
                st.markdown(meta["when_its_useful"])

                st.markdown("**Prompts used (verbatim)**")
                for role, prompt in meta["prompts"].items():
                    render_prompt_block(role, prompt)

                st.markdown("**Metrics**")
                row = combined.loc[[key]].copy()
                for metric in row.columns:
                    row[metric] = row[metric].map(lambda v, m=metric: format_metric(m, v))
                st.dataframe(row.rename(columns=METRIC_LABELS), width="stretch")

        st.markdown("### Evaluation metrics, defined")
        for term, definition in TECHNIQUE_METRIC_TERMINOLOGY.items():
            with st.expander(term):
                st.markdown(definition)

    # ---- Document Explorer --------------------------------------------------
    with tabs[4]:
        st.header("Document Explorer")
        st.markdown("See every technique's answer to the same document, side by side.")

        document_ids = sorted(records["document_id"].unique().tolist())
        document_lookup = records.drop_duplicates("document_id").set_index("document_id")
        document_id = st.selectbox(
            "Document",
            options=document_ids,
            format_func=lambda d: f"{d} — {document_lookup.loc[d, 'length']}/{document_lookup.loc[d, 'position']}",
        )
        selected = document_lookup.loc[document_id]
        st.markdown(f"**Question:** {selected['question']}")
        st.markdown(f"**Gold answer:** {selected['gold_answer']}")
        st.markdown(f"**Length:** {LENGTH_LABELS.get(selected['length'], selected['length'])} · **Position:** {POSITION_LABELS.get(selected['position'], selected['position'])}")

        document_records = records[records["document_id"] == document_id].set_index("technique")
        for key, meta in techniques_meta.items():
            if key not in document_records.index:
                continue
            row = document_records.loc[key]
            with st.expander(f"{meta['name']} — predicted: {row['predicted_answer']}", expanded=False):
                correct = format_metric("f1", row["f1"])
                st.markdown(f"**F1:** {correct} · **LLM calls:** {row['llm_calls']} · **Wall-clock:** {row['wall_clock_seconds']:.1f}s")
                st.markdown("**Context actually sent to the final answer call:**")
                st.text_area("context", row["context_sent_to_generator"], height=150, key=f"context_{key}", label_visibility="collapsed")
                st.markdown("**Steps**")
                render_step_trace(row["steps"])


if __name__ == "__main__":
    main()
