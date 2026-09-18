"""Streamlit dashboard: methodology, a per-shape deep dive (verbatim
prompts, a full step-by-step trace, and metrics scoped to what actually
applies to that shape), and a reference tab (caveats, what this task does
and doesn't test, future work, terminology, research).

Metrics are compared across shapes only where the comparison is
meaningful. A blanket "rank 1 of 4" table would rank plain LLM's 0%
handoff count as tied-best, as if it were succeeding at coordination
rather than having no second role to coordinate with at all. See
METRIC_APPLIES_TO below.
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

ALL_SHAPES = {"plain_llm", "single_shot_rag", "single_agent_react", "multi_agent_supervisor"}
TOOL_USING_SHAPES = {"single_shot_rag", "single_agent_react", "multi_agent_supervisor"}
LOOPING_SHAPES = {"single_agent_react", "multi_agent_supervisor"}

# Which shapes a metric is structurally meaningful for. A shape missing
# from a metric's set isn't scored low on it here, the metric doesn't
# apply: e.g. handoff count is 0 for every shape but multi-agent, not
# because the other three coordinate poorly, but because they have only
# one role to begin with. Mirrors EVALUATION_CRITERIA.md's Critical
# Metrics table, scoped to the metrics this experiment actually computes.
METRIC_APPLIES_TO = {
    "exact_match": ALL_SHAPES,
    "f1": ALL_SHAPES,
    "bridge_f1": ALL_SHAPES,
    "comparison_f1": ALL_SHAPES,
    "mean_step_count": ALL_SHAPES,
    "mean_tool_calls": TOOL_USING_SHAPES,
    "tool_error_rate": TOOL_USING_SHAPES,
    "mean_handoffs": {"multi_agent_supervisor"},
    "early_termination_rate": LOOPING_SHAPES,
    "redundant_step_rate": LOOPING_SHAPES,
    "mean_state_overhead_bytes": set(),  # tracked per shape, not comparable across shapes as measured here, see Caveats
    "mean_wall_clock_seconds": ALL_SHAPES,
    "mean_prompt_tokens": ALL_SHAPES,
    "mean_completion_tokens": ALL_SHAPES,
    "mean_latency_seconds": ALL_SHAPES,
    "mean_time_to_first_token_seconds": ALL_SHAPES,
    "mean_context_payload_bytes": ALL_SHAPES,
    "empty_retrieval_rate": TOOL_USING_SHAPES,
    "error_rate": ALL_SHAPES,
}
NOT_APPLICABLE_REASON = {
    "mean_tool_calls": "no tool calls in this shape",
    "tool_error_rate": "no tool calls in this shape",
    "empty_retrieval_rate": "no tool calls in this shape",
    "mean_handoffs": "only one role in this shape",
    "early_termination_rate": "no loop; always completes in one pass",
    "redundant_step_rate": "no loop; always completes in one pass",
    "mean_state_overhead_bytes": "measured differently per shape, not comparable (see Caveats)",
}
METRIC_GROUPS = [
    ("Quality", ["exact_match", "f1", "bridge_f1", "comparison_f1"]),
    (
        "Agentic",
        [
            "mean_step_count", "mean_tool_calls", "tool_error_rate", "mean_handoffs",
            "early_termination_rate", "redundant_step_rate", "mean_state_overhead_bytes",
        ],
    ),
    (
        "Operational",
        [
            "mean_wall_clock_seconds", "mean_prompt_tokens", "mean_completion_tokens",
            "mean_latency_seconds", "mean_time_to_first_token_seconds",
            "mean_context_payload_bytes", "empty_retrieval_rate", "error_rate",
        ],
    ),
]

st.set_page_config(page_title="Evaluation Criteria in Practice", layout="wide")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame]:
    payload = json.loads(RESULTS_PATH.read_text())
    return payload["shapes"], pd.DataFrame(payload["runs"])


def combined_metrics(records: pd.DataFrame) -> pd.DataFrame:
    return quality_summary(records).join(agentic_summary(records)).join(operational_summary(records))


def format_metric(metric: str, value: float) -> str:
    return METRIC_FORMATS.get(metric, "{:.2f}").format(value)


def shape_metric_rows(combined: pd.DataFrame, shape_key: str, metrics: list[str], shapes_meta: dict) -> list[dict]:
    """One row per metric in this group, scoped to the shapes it actually
    applies to. A shape this metric doesn't apply to gets an explicit
    'Not applicable' row with why, never a misleading number or rank.
    Rank is spelled out as best/worst, and 'Compared against' names the
    other shapes by their real names, not their internal keys, so this
    table can be read on its own without decoding either."""
    rows = []
    for metric in metrics:
        applies_to = METRIC_APPLIES_TO[metric]
        label = METRIC_LABELS[metric]
        if shape_key not in applies_to:
            rows.append({"Metric": label, "Value": "Not applicable", "Rank": "—", "Compared against": NOT_APPLICABLE_REASON[metric]})
            continue
        value = combined.loc[shape_key, metric]
        if pd.isna(value):
            continue
        if not applies_to:
            rows.append({"Metric": label, "Value": format_metric(metric, value), "Rank": "—", "Compared against": "not comparable across shapes, see Caveats"})
            continue
        comparable = combined.loc[list(applies_to & set(combined.index)), metric].dropna()
        if len(comparable) < 2:
            rows.append({"Metric": label, "Value": format_metric(metric, value), "Rank": "—", "Compared against": "no other shape this applies to"})
            continue
        ascending = metric in LOWER_IS_BETTER
        rank = int(comparable.rank(ascending=ascending, method="min")[shape_key])
        total = len(comparable)
        if rank == 1:
            rank_label = f"{rank} of {total} (best)"
        elif rank == total:
            rank_label = f"{rank} of {total} (worst)"
        else:
            rank_label = f"{rank} of {total}"
        other_shapes = sorted(shapes_meta[s]["name"] for s in applies_to if s != shape_key)
        rows.append(
            {
                "Metric": label,
                "Value": format_metric(metric, value),
                "Rank": rank_label,
                "Compared against": ", ".join(other_shapes),
            }
        )
    return rows


def render_mermaid(diagram: str, height: int = 220) -> None:
    # This experiment's diagrams live inside st.expander sections, collapsed
    # by default and only mounted when opened, exactly the late-visibility
    # scenario that broke a simpler width-poll-then-render-once approach
    # elsewhere in this repo (see rag-architectures/app.py, where this fix
    # was worked out by inspecting broken renders directly): mermaid.run()'s
    # promise can resolve before its own layout pass has actually finished,
    # so checking or retrying immediately reads a transiently broken SVG, or
    # collides with the still-finishing first pass and corrupts it further.
    # Fixed the same way here: poll width with setTimeout (keeps firing even
    # while hidden, unlike requestAnimationFrame), enforce a minimum real
    # time floor on top of width stability, then render once, wait a fixed
    # generous delay, and only retry (at most once) if the result is still
    # degenerate.
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

tab_methodology, tab_deep_dive, tab_decision_framework, tab_reference = st.tabs(
    ["Methodology", "Deep Dive", "Decision Framework", "Reference & Future Work"]
)

with tab_methodology:
    st.header("Goal")
    st.markdown(
        "This experiment holds the task and dataset fixed and varies the *shape itself*: "
        "plain LLM call, RAG, single agent, multi-agent, to test the central claim of "
        "[EVALUATION_CRITERIA.md](../../EVALUATION_CRITERIA.md): that a task/quality "
        "metric alone under-determines outcomes more as a system's shape gains more decision "
        "points, and that the metric layer that actually explains a close call is whichever "
        "one is specific to what that shape can uniquely get wrong."
    )

    st.header("The problem: multi-hop question answering")
    st.markdown(
        f"**[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)** (distractor "
        "config): every question needs facts from two different paragraphs to answer, and "
        "every question ships 10 candidate paragraphs, 2 gold and 8 distractors, forcing a "
        "real test of whether a shape can find and use more than one fact. The `search()` "
        "tool returns only its single best-matching paragraph (k=1) and is identical across "
        "every shape here, so any difference in results comes from what each shape can "
        "*decide to do* with that tool, not from a different or better retriever.\n\n"
        f"Every shape answers the same **{records['question_id'].nunique()} sampled "
        "questions**, split evenly between **bridge** (hop 2 needs an entity hop 1 finds) "
        "and **comparison** (two independent lookups compared against each other)."
    )

    st.header("Shapes compared")
    for meta in shapes_meta.values():
        st.markdown(f"**{meta['name']}:** {meta['description']}")
    st.caption("One representative technique per shape, chosen for being the strongest real contender for that shape, not a strawman.")

    st.header("Metrics, and the one this experiment adds")
    st.markdown(
        "The same three categories tracked throughout this repo (quality, agentic, "
        "operational), plus **redundant step rate**: a later tool call retrieving a "
        "paragraph title an earlier tool call in the same run already retrieved. This "
        "operationalizes [MAST](https://arxiv.org/abs/2503.13657)'s most common documented "
        "multi-agent failure mode, *step repetition* (17.14% of failures in that paper's "
        "study), concretely enough to compute from a run's own trace. It's a narrower, "
        "single-tool operationalization of \"redoing work already done\", not every way a "
        "system can repeat itself; see [EVALUATION_CRITERIA.md](../../EVALUATION_CRITERIA.md) "
        "for the gap this metric was built to close, and the Results tab for whether it "
        "actually caught anything in this run."
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
        "questions, so the metric wouldn't mean anything here."
    )

    st.header("Not every metric applies to every shape")
    st.markdown(
        "A metric that doesn't apply to a shape (handoff count for a single-role shape, "
        "early termination rate for a shape with no loop) is reported as **not applicable** "
        "in the Deep Dive tab, not as a 0 or a tied-best rank. Comparisons and rankings there "
        "are scoped to only the shapes a metric actually applies to."
    )

    st.header("Model backend")
    st.markdown(
        "Every model call in every shape goes through the same local "
        "[Ollama](https://ollama.com) server running `llama3.1:8b` (see `llm_client.py`), "
        "so differences in the results come from shape, not from which model answered."
    )

with tab_deep_dive:
    st.markdown(
        "Each shape on its own: what its diagram actually shows, what its own numbers say "
        "once metrics are scoped to what applies to it, and a real trace of it answering a "
        "real question. Cross-shape reasoning lives in the Decision Framework tab; a "
        "chart-based summary of all four shapes together sits at the end of this tab."
    )

    DIAGRAM_DESCRIPTIONS = {
        "plain_llm": (
            "What this diagram shows: the question goes straight to the model, which "
            "answers from training knowledge alone. No box in between, because there is "
            "no lookup step at all."
        ),
        "single_shot_rag": (
            "What this diagram shows: exactly two steps, one search against the corpus, "
            "then one generation call using whatever that search returned. There's no "
            "path back to the search box, this shape never gets a second try."
        ),
        "single_agent_react": (
            "What this diagram shows: the agent loops between reasoning and searching "
            "(the arrow back into the Agent box) until it either commits to an answer or "
            "the dotted path fires, it hits its turn cap with nothing committed."
        ),
        "multi_agent_supervisor": (
            "What this diagram shows: the Supervisor routes a query to the Retriever, "
            "whose result the Reasoner drafts an answer from; the Verifier either accepts "
            "that draft (solid arrow to the answer) or sends it back to the Supervisor to "
            "refine the query (labeled arrow), looping until accepted or the dotted "
            "round-cap path fires instead."
        ),
    }

    SHAPE_ANALYSIS = {
        "plain_llm": (
            "No metric here needs a rank to interpret: with no retrieval and no loop, this "
            "shape's entire outcome is decided by one generation call. Its bridge F1 "
            "(13.1%) against its comparison F1 (61.1%), a 4.7x gap, is the cleanest "
            "evidence in this experiment of what retrieval actually buys: on questions "
            "answerable in part from general knowledge, it does respectably; on questions "
            "needing a specific looked-up fact, it has no way to get there at all."
        ),
        "single_shot_rag": (
            "This shape ties the multi-agent shape on exact match (60.0%) while making "
            "one tool call and zero handoffs. Its only search found a gold paragraph 92.5% "
            "of the time (37 of 40), better than Single-Agent ReAct's own first search "
            "(82.5%) despite that shape getting to try again afterward. At 195 mean prompt "
            "tokens and 1.1s mean wall-clock, nothing in this experiment shows a cost this "
            "shape paid to reach that accuracy."
        ),
        "single_agent_react": (
            "This shape's early termination rate (35.0%) undersells how binary its "
            "outcomes actually are. On the 26 of 40 runs where it committed to an answer "
            "at all, it was right 73.1% of the time; on the other 14, it produced no "
            "answer, an automatic miss with nothing in between. Its redundant step rate "
            "(28.3%) means more than a quarter of its tool calls re-retrieved a paragraph "
            "it already had, evidence that its turn-by-turn search decisions aren't always "
            "purposeful, even on runs that ultimately succeeded."
        ),
        "multi_agent_supervisor": (
            "Splitting this shape's 40 runs by whether the Verifier ever approved a draft "
            "within its 3-round budget tells almost the whole story: the 21 that did score "
            "66.7% exact match, and not one of them shows a redundant retrieval. The 19 "
            "that didn't score 52.6% (still an unverified best-effort draft, not a blank), "
            "and every single one shows at least one redundant retrieval, its query "
            "refinement had run out of new things to try before its budget ran out."
        ),
    }

    diagram_heights = {
        "plain_llm": 120,
        "single_shot_rag": 140,
        "single_agent_react": 280,
        "multi_agent_supervisor": 200,
    }
    st.caption(
        "How to read the metrics tables below: **Value** is that shape's own number. "
        "**Rank** says where that number lands against the *other shapes this metric "
        "actually applies to*, labeled best or worst, not against all four shapes "
        "regardless of fit. **Compared against** names exactly which other shapes that "
        "rank is measured against. A metric can be **Not applicable**, meaning it doesn't "
        "apply to this shape at all (a reason is given), which is different from this "
        "shape scoring badly on it."
    )

    for shape_key in SHAPES.keys():
        meta = shapes_meta[shape_key]
        with st.expander(meta["name"]):
            render_mermaid(meta["diagram"], height=diagram_heights[shape_key])
            st.caption(DIAGRAM_DESCRIPTIONS[shape_key])

            st.markdown("#### What this shape's own numbers show")
            st.markdown(SHAPE_ANALYSIS[shape_key])

            st.markdown("#### How we implemented it")
            st.markdown(meta["how_we_implemented_it"])

            st.markdown("#### Metrics that apply to this shape")
            for group_label, metrics in METRIC_GROUPS:
                rows = shape_metric_rows(combined, shape_key, metrics, shapes_meta)
                if not rows:
                    continue
                st.markdown(f"**{group_label}**")
                st.dataframe(
                    pd.DataFrame(rows).set_index("Metric")[["Value", "Rank", "Compared against"]],
                    width="stretch",
                )

    st.divider()
    st.header("All shapes at a glance")
    st.markdown(
        "Single-shot RAG tied the winning multi-agent shape on accuracy (60.0% exact match "
        "each) and clearly beat the looping single-agent shape (47.5%), despite doing far "
        "less work: one search, one generation call, no loop, no verification. Plain LLM, "
        "with no retrieval at all, trailed every other shape (35.0%), the clearest evidence "
        "here that retrieval closes a real gap. The per-shape sections above explain the "
        "mechanism behind it; so does the Decision Framework tab."
    )
    quality = quality_summary(records)
    agentic = agentic_summary(records)
    st.dataframe(quality.rename(columns=METRIC_LABELS).style.format("{:.1%}"), width="stretch")
    st.bar_chart(quality[["exact_match", "f1", "bridge_f1", "comparison_f1"]])

    st.subheader("Accuracy vs. wall-clock time")
    st.caption(
        "RAG reaches the same accuracy as Multi-Agent for a fraction of the wall-clock "
        "cost; Single-Agent ReAct spends more time than either for a worse result. Cost "
        "alone doesn't explain this gap, see question 4 in the Decision Framework tab."
    )
    tradeoff = quality[["exact_match"]].join(agentic[["mean_wall_clock_seconds"]])
    st.scatter_chart(tradeoff, x="mean_wall_clock_seconds", y="exact_match")

with tab_decision_framework:
    st.caption(
        "The same six questions from [EVALUATION_CRITERIA.md](../../EVALUATION_CRITERIA.md)'s "
        "decision framework, this time answered with this experiment's own numbers instead of "
        "the three experiments that originally motivated the framework."
    )

    st.markdown("**1. Is the output good?**")
    st.markdown(
        "*Applies to:* all four shapes.\n\n"
        "*This experiment's answer:* RAG and Multi-Agent tied at 60.0% exact match; "
        "Single-Agent ReAct scored 47.5%; Plain LLM scored 35.0%. Read alone, this says "
        "\"shape doesn't matter above single-shot retrieval,\" true as far as it goes, but "
        "it can't explain why two shapes with such different machinery (RAG: 1 tool call, 0 "
        "handoffs, 1.1s; Multi-Agent: 2 tool calls, 7 handoffs, 9.5s) land in the same place. "
        "The next few questions answer that."
    )

    st.markdown("**2. If quality is tied or the gap looks small, why?**")
    st.markdown(
        "*Applies to:* RAG, Single-Agent ReAct, Multi-Agent. Not Plain LLM: with only one "
        "decision point, there's no further layer left to check once question 1 is answered.\n\n"
        "*This experiment's answer:* RAG's single search, using the raw question verbatim, "
        "found a gold paragraph 92.5% of the time (37 of 40), better than Single-Agent "
        "ReAct's own first, self-generated search (82.5%, 33 of 40). That's why RAG matched "
        "or beat the looping single-agent shape despite doing far less work: its retrieval "
        "was already better on the one try it got. Against Multi-Agent specifically, which "
        "tied RAG on quality, the next layer isn't retrieval quality, it's agentic metrics: "
        "see questions 3 and 6 below."
    )

    st.markdown("**3. Is the system's autonomy trustworthy, or is an external cap doing the work?**")
    st.markdown(
        "*In plain terms:* when a looping shape stops, is that because it judged it had "
        "enough information, or because we told it \"stop after N tries\" and it simply ran "
        "out of tries? If the cap is doing the work, whatever accuracy the shape gets is "
        "partly an accident of whichever limit we happened to pick, not evidence it can be "
        "trusted to know when it's actually done.\n\n"
        "*Applies to:* Single-Agent ReAct and Multi-Agent only. Not applicable to Plain LLM "
        "or RAG, confirmed directly rather than assumed: both measured exactly 0% early "
        "termination, not because they reliably decide to stop, but because neither has a "
        "stopping decision to make at all.\n\n"
        "**Verdict: partially trustworthy in both shapes when the self-stop signal actually "
        "fires, but neither can be relied on to always produce that signal, and what a "
        "shape does in its absence turns out to matter as much as the judgment itself.**\n\n"
        "*This experiment's answer:* Single-Agent ReAct reaches its own decision to stop "
        "(emits `finish[...]`) on 65% of runs (26 of 40), and is right 73.1% of the time "
        "when it does, clearly better than its 47.5% overall accuracy, a real signal. "
        "Multi-Agent's Verifier approves a draft within budget on 52.5% of runs (21 of 40), "
        "right 66.7% of the time when it does, also better than its 60.0% overall accuracy, "
        "also a real signal. Neither shape's self-judgment is dramatically more trustworthy "
        "than the other's when it actually fires. The difference that decides the overall "
        "accuracy gap is what happens the rest of the time, when the judgment never fires "
        "at all: Single-Agent ReAct's remaining 35% of runs (14 of 40) simply run out of "
        "turns and submit no answer, a guaranteed miss with the turn cap alone ending the "
        "run. Multi-Agent's remaining 47.5% (19 of 40) exhaust the round cap too, but still "
        "fall back to an unverified draft, right 52.6% of the time, well above chance. So "
        "the more trustworthy shape here isn't the one with better judgment, it's the one "
        "that fails more gracefully when its judgment doesn't fire at all. In both shapes, "
        "the external cap is doing real work on over a third of all runs, not a rare edge "
        "case, so overall accuracy is partly a property of the turn and round limits chosen "
        "(4 and 3), not solely of either shape's own judgment."
    )

    st.markdown("**4. Where does the cost come from, and is it buying anything?**")
    st.markdown(
        "*Applies to:* all four shapes.\n\n"
        "*This experiment's answer:* Multi-Agent's 9.5s mean wall-clock, 7 handoffs, and "
        "788 prompt tokens bought a tied accuracy with RAG's 1.1s, 0 handoffs, and 195 "
        "tokens. On this task, at this sample size, that extra machinery paid for "
        "redundancy (see question 6 below), not for accuracy RAG didn't already have."
    )

    st.markdown("**5. Is the system reliable, or just accurate on average?**")
    st.markdown(
        "*Applies to, with a caveat:* model-call error rate, all four shapes; tool error "
        "rate and empty retrieval rate, only the three tool-using shapes. Abstention rate "
        "isn't tracked at all in this experiment: none of these four shapes' prompts gave "
        "the model the option to decline an answer, so there's nothing to measure here, a "
        "gap in the prompts, not a property of the shapes.\n\n"
        "*This experiment's answer:* 0% model-call error rate and 0% tool error rate "
        "across every shape and every run. That's not a wasted metric, it's what lets "
        "every other finding here be read as a genuine behavioral difference rather than "
        "one shape just breaking more often. It also means this experiment, like the "
        "framework it tests, still hasn't stress-tested reliability under conditions where "
        "it would actually vary."
    )

    st.markdown("**6. Is coordination between roles adding value, or just adding cost and failure surface?**")
    st.markdown(
        "*Applies to:* Multi-Agent only, the one shape here with more than one role.\n\n"
        "*This experiment's answer:* with only one multi-agent representative, there's no "
        "other multi-agent architecture here to compare handoff count against. Splitting "
        "this shape's own 40 runs by outcome tells the same kind of story a cross-"
        "architecture comparison would: the 21 runs that settled within budget (zero "
        "redundant retrievals) scored 66.7% exact match; the 19 that exhausted their full "
        "3-round budget (every one with at least one redundant retrieval) scored 52.6%. "
        "More handoffs here didn't mean better coordination, it was the signature of "
        "coordination that had already stopped finding anything new."
    )

with tab_reference:
    st.header("Caveats")
    st.markdown(
        "- 40 questions is enough to see clear directional differences between shapes, not "
        "enough for tight statistical confidence on exact percentage-point gaps, especially "
        "once broken down further by question type.\n"
        "- A local 8B model at temperature 0 with strict output-format instructions is a "
        "noisier narrator than a larger hosted model would be; some of the gap between "
        "shapes may partly reflect how reliably `llama3.1:8b` follows a given prompt's "
        "format, not purely shape.\n"
        "- Redundant step rate only catches a literal duplicate paragraph retrieval. A shape "
        "that asks a differently-worded but substantively redundant question, without ever "
        "retrieving the same title twice, would not be caught by it.\n"
        "- State overhead isn't measured consistently enough across shapes to compare "
        "directly (see the Deep Dive tab's per-shape metrics); it's reported but not used in "
        "any finding."
    )

    st.header("What this task does, and doesn't, test")
    st.markdown(
        "The plan going in was that HotpotQA's multi-hop structure would be a generous test "
        "for this experiment's question: a shape that can't retrieve more than one fact "
        "(plain LLM, single-shot RAG) should be structurally capped in a way the task is "
        "built to expose. That held for plain LLM, but not for single-shot RAG: it matched "
        "or slightly beat the looping shapes even on bridge questions specifically, because "
        "a real share of HotpotQA's \"multi-hop\" questions don't actually need both "
        "supporting paragraphs to answer, a shortcut "
        "[Min et al. (2019)](https://arxiv.org/abs/1906.02900) already documented for this "
        "exact dataset. A fairer test of \"does a shape need more than one retrieval\" would "
        "need a dataset where that shortcut isn't available.\n\n"
        "It's also a poor fit for testing shape differences that aren't about *how many "
        "facts* a task needs: long-horizon planning with no clear stopping condition, or "
        "genuinely conflicting evidence a Verifier role would need to adjudicate rather than "
        "just reject, would stress the single-agent and multi-agent shapes in ways this "
        "task's fixed, mostly-shortcut-able ~2-hop structure doesn't."
    )

    st.header("Future work")
    st.markdown(
        "- **A dataset without HotpotQA's single-hop shortcut**, where a question genuinely "
        "cannot be answered from either supporting fact alone.\n"
        "- **A task with a wider, unknown range of required steps**, relevant to whether "
        "RAG's single-shot ceiling and the plain LLM's total lack of grounding scale the "
        "same way as step count grows.\n"
        "- **A hosted or batching-capable backend**, to separate what's a property of shape "
        "from what's a property of `llama3.1:8b` specifically following these prompts.\n"
        "- **A redundant-step rate that catches semantic, not just literal, repetition**, "
        "using an LLM-judge or embedding-similarity check on sub-question wording.\n"
        "- **The next experiment already queued in "
        "[EVALUATION_CRITERIA.md](../../EVALUATION_CRITERIA.md):** a task or corpus "
        "difficult enough to move reliability metrics off the 0%-ish floor they've sat at "
        "throughout this experiment."
    )

    st.header("Terminology")
    st.markdown(
        "**Shape.** How many decision points a system has: plain LLM has one (what to "
        "output), RAG has two (what to retrieve, what to generate), a single agent has as "
        "many as it takes steps, multi-agent has that many again, times however many roles "
        "hand off to each other.\n\n"
        "**Handoff.** One transfer of control, and whatever state was written into it, from "
        "one role to another. Always 0 for a single-role shape, not a low score, an "
        "inapplicable one.\n\n"
        "**Early termination.** A run that hit its step cap without the shape itself "
        "signaling it was done. Only meaningful for a shape with a loop to hit a cap on.\n\n"
        "**Redundant step rate.** How often a tool call retrieves a paragraph title an "
        "earlier tool call in the same run already retrieved, as a fraction of total tool "
        "calls.\n\n"
        "**Exact Match / F1.** Exact Match is strict (1 if the normalized strings match "
        "exactly, 0 otherwise); F1 gives partial credit for word overlap."
    )

    st.header("Grounding research")
    st.markdown(
        "* [Yao et al., 2022, ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)\n"
        "* [Lewis et al., 2020, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)\n"
        "* [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)\n"
        "* [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)\n"
        "* [Cemri et al., 2025, Why Do Multi-Agent LLM Systems Fail? (MAST)](https://arxiv.org/abs/2503.13657)\n"
        "* [Yang et al., 2018, HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)\n"
        "* [Min et al., 2019, Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://arxiv.org/abs/1906.02900)\n"
        "* This repo's own [EVALUATION_CRITERIA.md](../../EVALUATION_CRITERIA.md), the framework document this experiment tests"
    )
