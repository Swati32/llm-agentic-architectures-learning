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
import streamlit.components.v1 as components

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

# Hand-picked from the actual results (see the Comparison tab's "Analysis" section for how
# these were found), not a representative sample: each one exists to make one specific finding
# concrete, in the architectures' own traces, rather than asking a reader to browse all 40
# questions to rediscover it themselves.
SPOTLIGHT_EXAMPLES = [
    {
        "question_id": "5adbf84555429947ff17387c",
        "title": "The planner never says it's done",
        "why": (
            "A clean 2-hop comparison question: the Planner had already answered both "
            "sub-questions it needed by round 2, then invented an unnecessary third question "
            "anyway rather than stopping. See the Comparison tab's first finding."
        ),
        "trace_architectures": ["orchestrator_sequential"],
    },
    {
        "question_id": "5abcd77755429965836004ce",
        "title": "Single-Agent ReAct gets stuck; the \"smarter\" adaptive orchestrator gets it wrong",
        "why": (
            "Single-Agent ReAct never commits to an answer and burns its whole turn budget "
            "searching instead. The adaptive Orchestrator does commit, but to the wrong answer. "
            "The fixed pipeline, which does no adapting at all, gets it right. See the "
            "Comparison tab's findings on Single-Agent ReAct's stopping problem and on fixed vs. "
            "adaptive decomposition."
        ),
        "trace_architectures": ["single_agent_react", "orchestrator_sequential", "sequential_pipeline"],
    },
    {
        "question_id": "5ae0120155429925eb1afbfb",
        "title": "No clean win: the fixed pipeline guesses wrong here, and adaptive planning doesn't reliably fix it either",
        "why": (
            "The fixed pipeline's blind upfront guess is wrong. The adaptive orchestrator "
            "reaches the right answer, but only after also exhausting its round budget without "
            "ever saying it was done. The verification loop gets stuck for the opposite reason: "
            "it never finds evidence to verify its draft against, in any of its 3 rounds. See "
            "the Comparison tab's findings on the verification loop's limits and on fixed vs. "
            "adaptive decomposition."
        ),
        "trace_architectures": ["sequential_pipeline", "orchestrator_sequential", "supervisor_verification"],
    },
    {
        "question_id": "5ae1847e55429920d52343ee",
        "title": "Verification catches a subtler failure than a wrong fact: answering the wrong question",
        "why": (
            "Four architectures correctly retrieve Liuzhou's area, then answer with that number "
            "instead of actually naming which city is bigger. Only the Supervisor's Verifier "
            "role catches that the draft doesn't actually answer the question asked. See the "
            "Comparison tab's \"Retrieval quality only improved where a role was actually built "
            "to improve it.\""
        ),
        "trace_architectures": ["sequential_pipeline", "supervisor_verification"],
    },
    {
        "question_id": "5ab9fe1255429939ce03dc40",
        "title": "Some bridge questions beat every architecture",
        "why": (
            "All five architectures give a different, wrong answer here. No amount of "
            "coordination structure fixes a reasoning chain the model gets wrong in the first "
            "place. See the Comparison tab's \"Comparison questions were easier than bridge "
            "questions for every architecture, not just some.\""
        ),
        "trace_architectures": ["single_agent_react", "supervisor_verification"],
    },
]

# Two hand-picked runs per architecture for its own Deep-Dive tab: one where it does what
# it's designed to do, one that shows the specific weakness its own metrics point to. Not a
# free picker over all 40 questions, same reasoning as SPOTLIGHT_EXAMPLES above: a raw dropdown
# doesn't tell a reader which of 40 traces is actually worth reading.
DEEP_DIVE_TRACE_EXAMPLES = {
    "single_agent_react": [
        {"question_id": "5ab2958a554299449642c911", "label": "Works cleanly: two searches, then commits"},
        {"question_id": "5abb76fa5542992ccd8e7f48", "label": "Gets stuck: keeps searching, never commits to an answer"},
    ],
    "sequential_pipeline": [
        {"question_id": "5ab2958a554299449642c911", "label": "Works cleanly: both hops land on the gold paragraph"},
        {"question_id": "5abb76fa5542992ccd8e7f48", "label": "Guesses wrong: the blind upfront decomposition misses"},
    ],
    "orchestrator_sequential": [
        {"question_id": "5adbf84555429947ff17387c", "label": "Gets the right answer, but keeps going after it already had enough"},
        {"question_id": "5abcd77755429965836004ce", "label": "Adaptive re-planning still lands on the wrong answer"},
    ],
    "orchestrator_parallel": [
        {"question_id": "5ab2958a554299449642c911", "label": "Works cleanly: identical to the fixed pipeline's answer"},
        {"question_id": "5abb76fa5542992ccd8e7f48", "label": "Guesses wrong: same blind decomposition, same miss"},
    ],
    "supervisor_verification": [
        {"question_id": "5adcd6705542992c1e3a2426", "label": "Verification loop working: first draft rejected, second draft accepted"},
        {"question_id": "5ae0120155429925eb1afbfb", "label": "Stuck: 3 rounds of refinement, never finds evidence to verify"},
    ],
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


def format_predicted(value) -> str:
    # A run that never committed to an answer (early terminated with no
    # finish/verdict) stores None, which pandas turns into float NaN once
    # it's in a DataFrame column, and "nan" reads like a bug rather than
    # the actual outcome.
    return "(no answer)" if pd.isna(value) else str(value)


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


def render_mermaid(diagram: str, height: int = 220) -> None:
    # The first time a tab is switched into, its newly-mounted iframe can
    # briefly have zero layout width while Streamlit is still sizing the
    # panel around it. mermaid's default `startOnLoad` renders immediately
    # on script load and measures labels against whatever width exists at
    # that instant, so a render that races the layout collapses to a tiny,
    # illegible diagram (this is reproducible: switching to a different
    # architecture afterwards, once the panel is already sized, renders
    # correctly first try). Disabling startOnLoad and polling for a real
    # width before calling mermaid.run() avoids racing that layout pass.
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
        "(`all-MiniLM-L6-v2`) for semantic similarity, not live web search. Every "
        "architecture is judged against exactly the same evidence.\n\n"
        f"Every architecture answers the same **{records['question_id'].nunique()} "
        "sampled questions**, split evenly between the two question types HotpotQA "
        "labels: **bridge** (hop 2 needs an entity hop 1 finds) and **comparison** "
        "(two independent lookups compared against each other). The `search()` tool "
        "returns only its single best-matching paragraph (k=1). With 2 gold "
        "paragraphs needed, one search can never fully answer a question, forcing "
        "every architecture to actually decide whether and how to search again."
    )

    st.header("Architectures compared")
    for meta in architectures_meta.values():
        st.markdown(f"**{meta['name']}:** {meta['description']}")

    st.header("Agentic metrics")
    st.markdown(
        "- **Step / loop count**: how many model or tool calls one question took\n"
        "- **Tool execution latency**: wall-clock time spent in `search()` itself\n"
        "- **Tool error / retry rate**: fraction of runs with at least one failed tool call\n"
        "- **Inter-agent handoff count**: how many times control passed between roles "
        "(planner, worker, verifier, ...). It's 0 for the single-agent baseline by definition\n"
        "- **Early termination rate**: fraction of runs that hit their step cap without "
        "a clean finish or verdict, rather than stopping because the architecture decided it was done\n"
        "- **State overhead**: size (bytes) of the shared state passed at the final handoff. "
        "For the single agent this is its whole running transcript instead"
    )

    st.header("Operational metrics")
    st.markdown(
        "- **Tokens used**, **context payload size**: cost drivers, and directly comparable here "
        "since every architecture runs on the same local model\n"
        "- **Latency** and **time to first token**: of the *first* model call in the run\n"
        "- **Empty retrieval rate**: redefined for this task. Since `search()` always returns its "
        "top-1 match, \"empty\" here means the retrieved paragraph wasn't one of the 2 gold "
        "paragraphs, a wasted or misleading lookup\n"
        "- **Error rate**: fraction of model calls that failed outright"
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
    quality = quality_summary(records)
    agentic = agentic_summary(records)
    operational = operational_summary(records)

    st.header("Quick reference: strongest and weakest metric")
    st.caption(
        "For each architecture, its top 2 metrics that actually distinguish it from the other "
        "four (Worked well) and its bottom 2 (Struggled). A metric every architecture is tied "
        "on, e.g. a 0% error rate across the board, is excluded: it isn't a distinguishing "
        "strength or weakness for any one of them. This table is a starting point for orientation, "
        "not the comparison. The full analysis, with the mechanism behind each result, is below."
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

    st.header("Analysis: what happened, and why")

    st.markdown("#### The adaptive planner never once decided it had enough information")
    st.markdown(
        "Orchestrator (sequential dispatch)'s early termination rate is **100%**: every one of "
        "the 40 runs used its full 3-round budget rather than the Planner ever emitting "
        "`Next: DONE`. Looking at actual planner output shows why concretely: for \"Are both "
        "Adolfo Bioy Casares and James Norman Hall Argentinian authors?\", a clean 2-hop "
        "comparison question, the Planner asked for Bioy Casares' nationality, then Hall's "
        "nationality (both answered after 2 rounds), then invented a third question anyway: "
        "\"What is the nationality of the co-author of James Norman Hall's novel "
        "'Mutiny on the Bounty'?\" It never judged that it was done; the step cap was the only "
        "thing that ever stopped it. This isn't a new discovery: it's the same problem "
        "ReAct-style agents were already known to have when a stopping condition depends on the "
        "model itself recognizing it has enough "
        "([Yao et al., 2022](https://arxiv.org/abs/2210.03629)). What this experiment adds is "
        "that it shows up in a dedicated Planner role too, not just a single agent's own loop, "
        "which is the next finding."
    )

    st.markdown("#### Single-Agent ReAct hit the identical failure in a different shape")
    st.markdown(
        "Single-Agent ReAct never emitted `Action: finish[...]` on **35%** of runs (14 of 40), "
        "instead looping on `search[...]` until it hit its 4-turn cap. But on the 26 runs where "
        "it did commit to an answer, it was right **73%** of the time (19 of 26). The model's "
        "reasoning wasn't the bottleneck here; deciding it had looked hard enough was. Put "
        "next to the previous finding, the same root cause shows up in two structurally "
        "different architectures: a single agent freely deciding whether to act again, and an "
        "orchestrator's dedicated Planner role deciding whether to delegate again. Both "
        "defaulted to \"look for more\" over \"commit to an answer\", which points at this being "
        "a property of the model's disposition toward the search-versus-finish decision itself, "
        "not of either control-flow structure."
    )

    st.markdown("#### The verification loop won on accuracy by rescuing bad retrievals, not by chaining hops")
    st.markdown(
        "Supervisor + Verification Loop reached **60%** exact match, 12.5 points above every "
        "other architecture, and its F1 lead is almost entirely on comparison questions "
        "(**0.836**, versus 0.561-0.676 elsewhere). Its lead evaporates on bridge questions, "
        "where its 0.410 F1 is statistically indistinguishable from the rest (0.389-0.435 on 20 "
        "questions). The round-count distribution shows the mechanism: 19 of 40 runs had their "
        "first draft accepted immediately, 19 of 40 exhausted all 3 rounds without ever getting "
        "the Verifier's approval, and only 2 landed in between. That bimodal split says query "
        "refinement works when the *first* retrieval was merely mediocre, giving the Supervisor "
        "room to phrase a better query against the same fixed corpus, but doesn't help when the "
        "corpus genuinely lacks better evidence for that phrasing, or when the missing piece is "
        "a second hop the loop was never designed to chase. This matches a broader finding that "
        "self-correction without new external information tends to plateau or thrash rather "
        "than converge ([Huang et al., 2023](https://arxiv.org/abs/2310.01798)): the Verifier "
        "can reject a bad draft, but \"try a different search query\" isn't new information "
        "when the same 10 paragraphs are all that exist."
    )

    st.markdown("#### Parallel dispatch bought almost nothing, and the reason is the serving backend")
    st.markdown(
        "Orchestrator (parallel dispatch) and Sequential Pipeline (fixed) share the exact same "
        "upfront decomposition and produced **byte-for-byte identical predicted answers** on "
        "every one of the 40 questions (both 47.5% exact match, both 0.527 F1, identical bridge "
        "and comparison splits). That's expected at temperature 0: dispatch order doesn't "
        "change what either hop retrieves or what the model outputs, only when it happens. The "
        "wall-clock gap is more informative: 5.1s parallel versus 5.7s sequential, about 10%, "
        "far short of the roughly 2x a truly concurrent backend should give two independent "
        "lookups. Ollama was running with a single parallel processing slot, so concurrent "
        "requests from the two worker threads were still served one at a time. The lesson "
        "generalizes: parallel dispatch is a property of your architecture's *intent*, but the "
        "latency win only shows up if your serving layer (batching, multiple GPUs, multiple "
        "hosted replicas) actually honors that intent."
    )

    st.markdown("#### Fixed decomposition guessed at bridge hops it couldn't have known, and adaptive planning didn't clearly fix it")
    st.markdown(
        "The original hypothesis was that Sequential Pipeline's blind, upfront decomposition "
        "would specifically hurt bridge questions, where hop 2 needs an entity hop 1 hasn't "
        "found yet, and that Orchestrator (sequential dispatch)'s adaptive re-planning would "
        "recover that gap. The data doesn't support a clean win: bridge F1 was **0.435** for "
        "the fixed pipeline versus **0.389** for the adaptive orchestrator, a difference well "
        "inside the noise of a 20-question sample, not the clear recovery the hypothesis "
        "predicted. Spot-checking individual traces shows why the adaptive version doesn't "
        "cleanly help even when it could: per the first finding above, its Planner keeps "
        "generating additional sub-questions past the point where it already had the bridge "
        "entity, and that extra, sometimes off-target searching dilutes the evidence handed to "
        "the Synthesizer as often as it sharpens it. Adaptive planning only pays for itself if "
        "the planner also knows when to stop adapting."
    )

    st.markdown("#### State overhead was Single-Agent ReAct's real cost, hidden inside \"just one agent\"")
    st.markdown(
        "Single-Agent ReAct carried a mean state overhead of **2,855 bytes** (its entire "
        "running transcript), 4 to 10 times every multi-agent architecture (292-762 bytes), "
        "because a multi-agent handoff only carries forward a small structured summary (a "
        "sub-question and its answer), while a single agent's context is its whole history, "
        "verbatim. That's also why it had the highest mean prompt tokens (1,317) and highest "
        "latency (10.9s) despite doing the least explicit coordination (0 handoffs, by "
        "construction): \"no coordination overhead\" doesn't mean \"no overhead\", it means the "
        "overhead moved into the transcript instead of into handoff messages."
    )

    st.subheader("Other observations")

    st.markdown("**Comparison questions were easier than bridge questions for every architecture, not just some.**")
    st.markdown(
        "Bridge F1 clusters tightly and low across all five architectures (0.389-0.435), while "
        "comparison F1 is both higher and more spread out (0.561-0.836). That gap holds "
        "regardless of architecture, which points at something structural about the task, not "
        "a specific control-flow weakness. A comparison question's two sub-lookups are usually "
        "self-contained (\"how tall is A\", \"how tall is B\"), so retrieval accuracy alone gets "
        "you most of the way. A bridge question needs the model to correctly name an "
        "intermediate entity, in natural language, before the second query can even be formed, "
        "an extra inferential step that fails independently of how good the retrieval is. The "
        "original HotpotQA paper draws exactly this distinction between the two question types "
        "([Yang et al., 2018](https://arxiv.org/abs/1809.09600))."
    )

    st.markdown("**More handoffs didn't mean better coordination.**")
    st.markdown(
        "Orchestrator (sequential dispatch) and Supervisor + Verification Loop both average 7 "
        "handoffs, the joint-highest of all five architectures, yet one is tied for the lowest "
        "accuracy (47.5%) and the other is the highest (60%). The amount of coordination on the "
        "agentic-metrics dashboard doesn't predict quality by itself; what those handoffs are "
        "*for* does. The orchestrator's handoffs are spent gathering more sub-questions, "
        "including unnecessary ones (see above). The supervisor's are spent checking and "
        "re-trying evidence. Same cost by this metric, different payoff."
    )

    st.markdown("**Token spend didn't track with accuracy either.**")
    st.markdown(
        "Orchestrator (sequential dispatch) spent the second-most prompt tokens per question "
        "(1,196), behind only Single-Agent ReAct (1,317), yet tied for the lowest exact match "
        "(47.5%) with two architectures that spent almost half as many tokens (643, both the "
        "fixed pipeline and the parallel dispatch). Supervisor, the accuracy winner, spent a "
        "moderate 788, less than both lower-accuracy, higher-token architectures. Spending more "
        "tokens on an architecture that keeps re-asking questions the model didn't actually "
        "need doesn't buy accuracy. Spending a bit more on a role whose entire job is checking "
        "whether an answer is trustworthy does."
    )

    st.markdown("**Retrieval quality only improved where a role was actually built to improve it.**")
    st.markdown(
        "Empty retrieval rate, the fraction of searches that miss both gold paragraphs, sits in "
        "a narrow band for four architectures (17.5-19.8%) and drops meaningfully only for "
        "Supervisor (9.6%), the one architecture whose queries can be revised based on explicit "
        "feedback about what evidence was missing. None of the other four ever rewrites a query "
        "after seeing it fail; Supervisor is the only one built to. That's a real, measurable "
        "benefit of the verification loop, distinct from (and smaller than) its overall "
        "accuracy win, and it's the mechanism behind why the accuracy win happens at all."
    )

    st.header("Supporting data")

    st.subheader("Quality: all architectures")
    st.dataframe(quality.rename(columns=METRIC_LABELS).style.format("{:.1%}"), width="stretch")
    st.bar_chart(quality[["exact_match", "f1", "bridge_f1", "comparison_f1"]])

    st.subheader("Agentic cost: all architectures")
    agentic_formats = {METRIC_LABELS[c]: METRIC_FORMATS.get(c, "{:.2f}") for c in agentic.columns}
    st.dataframe(agentic.rename(columns=METRIC_LABELS).style.format(agentic_formats), width="stretch")
    st.bar_chart(agentic[["mean_step_count", "mean_handoffs"]])
    st.bar_chart(agentic[["early_termination_rate", "tool_error_rate"]])

    st.subheader("Operational cost: all architectures")
    operational_formats = {METRIC_LABELS[c]: METRIC_FORMATS.get(c, "{:.2f}") for c in operational.columns}
    st.dataframe(operational.rename(columns=METRIC_LABELS).style.format(operational_formats), width="stretch")
    st.bar_chart(operational[["mean_latency_seconds"]])
    st.bar_chart(operational[["mean_prompt_tokens", "mean_completion_tokens"]])

    st.subheader("Accuracy vs. wall-clock time")
    tradeoff = quality[["exact_match"]].join(agentic[["mean_wall_clock_seconds"]])
    st.scatter_chart(tradeoff, x="mean_wall_clock_seconds", y="exact_match")

    st.caption(
        "40 questions is enough to see clear directional differences, not enough for tight "
        "statistical confidence on the exact percentage-point gaps. Treat any single-digit "
        "difference between architectures as noise; the findings above are stated at the "
        "confidence the underlying traces actually support."
    )

with tab_deep_dive:
    architecture_key = st.selectbox(
        "Architecture", options=list(ARCHITECTURES.keys()), format_func=lambda k: architectures_meta[k]["name"]
    )
    meta = architectures_meta[architecture_key]
    st.subheader(meta["name"])

    st.markdown("#### What it is")
    st.markdown(meta["what_it_is"])

    # Measured from each diagram's actual rendered viewBox height, plus
    # margin, rather than guessed: flowchart height depends on how many
    # parallel lanes mermaid lays a diagram's branches into, not on how
    # many nodes or loops it has, so a per-architecture value beats one
    # constant for every diagram.
    diagram_heights = {
        "single_agent_react": 280,
        "sequential_pipeline": 110,
        "orchestrator_sequential": 210,
        "orchestrator_parallel": 200,
        "supervisor_verification": 200,
    }
    render_mermaid(meta["diagram"], height=diagram_heights[architecture_key])

    st.markdown("#### How we implemented it")
    st.markdown(meta["how_we_implemented_it"])

    st.markdown("#### When it's useful")
    st.markdown(meta["when_its_useful"])

    st.markdown("#### How it ranks against the other four architectures, best to worst")
    ranked = ranked_metrics(combined, architecture_key)
    ranked_table = pd.DataFrame(
        [{"Metric": METRIC_LABELS[m], "Value": format_metric(m, v), "Rank": f"{r} of {n}"} for m, v, r, n in ranked]
    )
    st.dataframe(ranked_table.set_index("Metric"), width="stretch")

    st.markdown("#### Prompts used, verbatim")
    for role, prompt in meta["prompts"].items():
        with st.expander(f"{role}"):
            st.text(prompt)

    st.markdown("#### Example runs")
    st.caption("One question this architecture handles the way it's meant to, and one that shows its characteristic weakness.")
    architecture_records = records[records["architecture"] == architecture_key]
    for example in DEEP_DIVE_TRACE_EXAMPLES[architecture_key]:
        example_run = architecture_records[architecture_records["question_id"] == example["question_id"]].iloc[0]
        with st.expander(f"{example['label']}: “{example_run['question']}”"):
            st.markdown(f"**Gold answer:** `{example_run['gold_answer']}` · **Predicted:** `{format_predicted(example_run['predicted_answer'])}` · **F1:** {example_run['f1']:.2f}")
            render_trace(example_run["steps"])

with tab_explorer:
    st.header("Spotlight examples")
    st.caption(
        "Not a browser over all 40 questions: these are the specific examples that most clearly "
        "show *why* the architectures behaved differently, each tied to a finding from the "
        "Architecture Comparison tab. Open one to see the actual traces behind the claim."
    )

    for spotlight in SPOTLIGHT_EXAMPLES:
        question_records = records[records["question_id"] == spotlight["question_id"]].set_index("architecture")
        gold_answer = question_records.iloc[0]["gold_answer"]
        question_type = question_records.iloc[0]["question_type"]
        question_text = question_records.iloc[0]["question"]

        with st.expander(spotlight["title"]):
            st.markdown(spotlight["why"])
            st.markdown(f"**Question:** {question_text}")
            st.markdown(f"**Gold answer:** `{gold_answer}` · **Type:** `{question_type}`")

            display_summary = pd.DataFrame(
                {
                    "Predicted": question_records["predicted_answer"].map(format_predicted),
                    "Correct": question_records["exact_match"].map({1.0: "✅", 0.0: "❌"}),
                    "Early terminated": question_records["early_terminated"].map({True: "yes", False: ""}),
                }
            ).rename(index=lambda k: architectures_meta[k]["name"])
            st.dataframe(display_summary, width="stretch")

            for architecture_key in spotlight["trace_architectures"]:
                st.markdown(f"**{architectures_meta[architecture_key]['name']}, full trace:**")
                render_trace(question_records.loc[architecture_key, "steps"])
