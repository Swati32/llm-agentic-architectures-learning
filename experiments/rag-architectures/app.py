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

# Mirrors the README's Terminology section, including the metrics shown in
# the Architecture Deep Dive tab's own Metrics table below, so a reader
# doesn't have to leave the dashboard to look up what a number means.
CONCEPT_TERMINOLOGY = {
    "Chunk": (
        "A piece of a longer document, split up so it's small enough to embed meaningfully and "
        "retrieve individually. A whole 70,000-character news article can't be usefully compared "
        "against a short query as one embedding; chunking is what makes retrieval granular enough "
        "to work at all."
    ),
    "Embedding": (
        "A list of numbers (a vector) that a model produces to represent the meaning of a piece of "
        "text, such that text with similar meaning ends up with similar-looking vectors, even if it "
        "shares no exact words. \"Who leads the company\" and \"chief executive officer\" can end up "
        "close together in embedding space despite sharing no words at all."
    ),
    "Dense retrieval (semantic search) vs. BM25": (
        "Two different, opposite ways to rank chunks against a query, and the two signals Hybrid "
        "RAG fuses together.\n\n"
        "**Dense retrieval** embeds the query and every candidate chunk as vectors, then ranks "
        "chunks by how close their vector is to the query's (cosine similarity). Its strength is "
        "meaning: it can match a paraphrase that shares no words with the source text. Its weakness "
        "is that same fuzziness: it can miss an exact name, date, or number that a keyword search "
        "would catch instantly.\n\n"
        "**BM25** is a classic, decades-old keyword-ranking function that scores a chunk against a "
        "query purely by term frequency, how often the query's actual words appear in the chunk, "
        "weighted by how rare those words are across the corpus. It has no notion of meaning, so it "
        "can't find a paraphrase, but it never misses an exact word that's actually written in the "
        "text.\n\n"
        "In short: dense retrieval finds a chunk that *means* the same thing; BM25 finds a chunk "
        "that *says* the same thing."
    ),
    "Reciprocal rank fusion (RRF)": (
        "A way to merge two separately-ranked lists (e.g. one from dense search, one from BM25) "
        "into a single ranking, using each item's *position* in each list rather than its raw "
        "score. This avoids the problem that a cosine similarity and a BM25 score live on "
        "completely different, incomparable numeric scales."
    ),
    "Cross-encoder": (
        "A model that scores a (query, passage) pair by reading both together in one forward pass, "
        "as opposed to a bi-encoder (what dense retrieval uses), which embeds the query and the "
        "passage independently and compares the two vectors afterward. A cross-encoder is more "
        "accurate because it can attend across the two texts directly, but far too slow to run over "
        "an entire corpus, which is why it's only ever used to rerank a shortlist a cheaper method "
        "narrowed first."
    ),
    "HyDE (Hypothetical Document Embeddings)": (
        "Retrieving using the embedding of a model-generated hypothetical answer instead of the "
        "embedding of the raw query, on the idea that a hypothetical answer is phrased more like "
        "the real supporting text than a question is."
    ),
    "Query decomposition": (
        "Splitting one question into several simpler, self-contained sub-questions, retrieving "
        "separately for each, and combining the results. Aimed at questions that are really more "
        "than one lookup wearing one sentence, like a comparison between two things."
    ),
    "Corrective RAG (CRAG)": (
        "Explicitly grading whether retrieved evidence is actually relevant before generating an "
        "answer from it, and taking a corrective action (retrieve again differently, or abstain) "
        "rather than trusting whatever the first retrieval pass returned."
    ),
}

# The metrics shown in the Comparison tab's tables, defined right there
# rather than in the Terminology tab, so a reader can look up what a
# column means without leaving the comparison they're actually looking at.
RAG_METRIC_TERMINOLOGY = {
    "Recall@k / Precision@k / MRR": (
        "Standard information-retrieval metrics. Recall@k: did a relevant (gold) item appear "
        "anywhere in the top k retrieved. Precision@k: what fraction of the top k retrieved were "
        "actually relevant. MRR: the reciprocal of the rank of the first relevant item (1/1 if "
        "it's first, 1/2 if second, etc.), averaged across queries, rewarding a relevant item "
        "appearing earlier over appearing later."
    ),
    "Faithfulness / groundedness": (
        "Whether an answer's claims are actually supported by the retrieved context, as opposed to "
        "being correct by coincidence (or memorized from training) while citing context that "
        "doesn't actually back it up. The gold-standard way to check this is an LLM judge comparing "
        "each claim in the answer against the context; this experiment uses a cheaper "
        "lexical-overlap proxy instead."
    ),
    "Exact Match / F1": (
        "Two ways of scoring a predicted answer against the correct one. Exact Match is strict: 1 "
        "if the (normalized) strings match exactly, 0 otherwise. F1 gives partial credit based on "
        "word overlap, so a predicted answer that's mostly right but missing or adding a word still "
        "scores above zero."
    ),
    "Abstention (correct / incorrect)": (
        "Whether an architecture said \"Insufficient information.\" instead of giving a real "
        "answer. Abstaining is *correct* on a null query, where the corpus genuinely has no answer, "
        "and *incorrect* on an answerable query, where it means the architecture gave up on a "
        "question it could have answered. The same behavior is a win in one case and a failure in "
        "the other, which is why this experiment tracks the two rates separately."
    ),
    "LLM calls / retrieval rounds": (
        "How many separate calls to the language model, and how many separate retrieval attempts, "
        "one query needed. HyDE and Query Decomposition each make 2 LLM calls per query (versus 1 "
        "for the single-pass architectures) because each needs an extra call before the final "
        "answer. A round counts a retry as a new attempt; a multi-query fan-out (several "
        "sub-queries searched at once) still counts as 1 round."
    ),
    "Prompt tokens / completion tokens": (
        "How much text, measured in tokens (roughly, word-pieces), a call sends to the model "
        "(prompt) and gets back (completion). Prompt tokens track with how many chunks and how big "
        "they are; completion tokens track with how much the model had to write."
    ),
    "Latency / wall-clock time": (
        "Latency is how long a single model call took, start to finish. Wall-clock time is how "
        "long the *whole* query took, including every retrieval call and every LLM call in the "
        "run, which is what actually matters to whoever is waiting on an answer."
    ),
    "Time to first token (TTFT)": (
        "How long a model takes to start responding, as opposed to latency, which includes the "
        "time to finish the whole response. It's the metric closest to how responsive a system "
        "*feels*, since a user is waiting on the first token, not the last one. This dashboard "
        "reports the *median* TTFT rather than the mean, because a local-serving artifact hit "
        "during this experiment's own run corrupted a handful of raw TTFT values badly enough to "
        "make the mean meaningless (see the README's \"Why these metrics\" section)."
    ),
    "Context payload bytes": (
        "The size, in bytes, of everything sent to the model in one call: the system prompt, the "
        "passages, the question. It's a proxy for how much the model has to read before it can "
        "answer, distinct from token count because it's measured before tokenization."
    ),
    "Empty retrieval rate": (
        "How often a retrieval call came back with literally zero chunks. With a shared corpus of "
        "thousands of chunks, a plain top-k search essentially never returns fewer than k results, "
        "so this metric mostly matters for a narrower, more specific search that could plausibly "
        "find nothing."
    ),
    "Error rate": (
        "How often a model call failed outright (a timeout, a malformed response, a dropped "
        "connection), as opposed to succeeding but giving a wrong or unhelpful answer."
    ),
}

# The chunking sub-experiment only tracks retrieval and generation quality
# (see run_chunking_experiment.py's build_record), not the operational
# metrics (tokens, latency, LLM calls) — chunking doesn't change how many
# calls a query makes, only what each call's retrieved context contains.
# So its definitions are a subset of RAG_METRIC_TERMINOLOGY, not a
# separate write-up, kept in sync by pulling from the same source text.
CHUNKING_METRIC_TERMINOLOGY = {
    key: RAG_METRIC_TERMINOLOGY[key]
    for key in ["Recall@k / Precision@k / MRR", "Exact Match / F1", "Faithfulness / groundedness", "Abstention (correct / incorrect)"]
}

# What each chunking strategy actually does, and why it's in this sweep at
# all: every strategy answers the same underlying question (how much
# surrounding context does one retrievable unit carry, and does the cut
# respect sentence/paragraph structure or ignore it) differently, which is
# what the Chunking Experiment tab's results are actually measuring the
# effect of. Keys match chunking/strategies.py's STRATEGIES dict.
CHUNKING_STRATEGY_INFO = {
    "fixed_128": {
        "label": "Fixed 128",
        "what_it_does": (
            "Cuts on raw word-count boundaries, every 128 words, with a small overlap, ignoring "
            "sentence or paragraph structure entirely. The fastest, simplest strategy to implement."
        ),
        "why_it_matters": (
            "A cut can land mid-sentence, so a fact that happens to straddle a chunk boundary ends "
            "up split across two chunks, complete in neither. Small chunks also mean a very focused "
            "embedding per chunk, one topic per vector, but more chunks overall to search."
        ),
    },
    "fixed_256": {
        "label": "Fixed 256",
        "what_it_does": (
            "The same raw word-count cut as Fixed 128, just twice the size per chunk (256 words), "
            "with a proportionally larger overlap."
        ),
        "why_it_matters": (
            "A midpoint in the fixed-size size sweep: still cuts mid-sentence with the same "
            "probability per cut, but each chunk now carries roughly double the surrounding context, "
            "testing whether more context per chunk helps or just dilutes the embedding."
        ),
    },
    "fixed_512": {
        "label": "Fixed 512",
        "what_it_does": (
            "The same raw word-count cut again, at 512 words per chunk, the largest fixed-size "
            "strategy in this sweep."
        ),
        "why_it_matters": (
            "The least likely fixed-size strategy to split a fact in half, simply because there's "
            "more room in each chunk, at the cost of packing more unrelated content into one "
            "embedding vector and giving the generator more text to read per chunk retrieved."
        ),
    },
    "sentence": {
        "label": "Sentence",
        "what_it_does": (
            "Packs whole sentences into a chunk, greedily, up to a target size, and starts a new "
            "chunk rather than ever cutting a sentence in half."
        ),
        "why_it_matters": (
            "Directly fixes the fixed-size strategies' main weakness: a fact-bearing sentence always "
            "stays whole in one chunk, so it's fully there for retrieval to find, not split across "
            "a boundary. Costs nothing extra to compute over a fixed-size cut."
        ),
    },
    "recursive": {
        "label": "Recursive",
        "what_it_does": (
            "Splits on paragraph breaks first; if a paragraph is still too big, falls back to "
            "sentences, then to fixed-size words as a last resort for one very long sentence. "
            "This experiment's default strategy for the main 6-architecture comparison."
        ),
        "why_it_matters": (
            "The most commonly reached-for strategy in real RAG pipelines (the LangChain-style "
            "'RecursiveCharacterTextSplitter' pattern): it respects a document's own structure "
            "(paragraphs) when that structure is well-behaved, and only degrades to a cruder cut "
            "when it has to."
        ),
    },
    "semantic": {
        "label": "Semantic",
        "what_it_does": (
            "Embeds every individual sentence, then cuts a new chunk wherever the similarity "
            "between two consecutive sentences' embeddings drops below a threshold, the idea being "
            "that a topic shift shows up as a dip in how similar two neighboring sentences are."
        ),
        "why_it_matters": (
            "The only strategy here that cuts on *meaning* rather than sentence/paragraph structure "
            "or a fixed size. It's also the most expensive to build, since it needs one embedding "
            "call per sentence (not per chunk) just to decide where the boundaries go, and it can "
            "produce very short, sometimes single-sentence chunks when the source text switches "
            "topics often."
        ),
    },
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
    # Three real bugs, found by inspecting broken renders directly rather
    # than guessing at timing:
    #
    # (1) mermaid.min.js auto-initializes itself the instant it loads,
    # using its own defaults, whenever document.readyState is already
    # "complete" — true here, since the script loads well after the page
    # has. Fixed structurally: the diagram source sits inertly in a
    # <script type="text/plain"> tag, invisible to mermaid, until this
    # code explicitly moves it into a `.mermaid` div and calls
    # mermaid.run() itself.
    # (2) The width-readiness poll used requestAnimationFrame, which can
    # be paused indefinitely for content that isn't currently visible
    # (Streamlit's tab panels, a backgrounded browser tab). Fixed by
    # polling with setTimeout instead, which keeps firing regardless.
    # (3) mermaid.run()'s returned promise can resolve *before* its own
    # internal layout pass has actually finished — checking the result
    # immediately in that promise's `.then()` sometimes reads a
    # transiently tiny SVG and (worse) retrying right away, before the
    # first pass has truly finished, collides with it and corrupts the
    # element's state entirely (confirmed directly: a manual, single,
    # unhurried call always renders correctly; an immediate reactive
    # retry sometimes does not). So the degeneracy check here waits a
    # fixed, generous delay *after* the call instead of reacting to the
    # promise, and retries at most once.
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
                // setTimeout, not requestAnimationFrame: see note (2) above.
                // Width stabilizing quickly is necessary but not sufficient —
                // observed directly: this environment's real paint/layout
                // settling can take noticeably longer than the width alone
                // suggests, and rendering the instant width looks stable
                // still produced a degenerate SVG. So on top of the width
                // check, also enforce a minimum floor of real elapsed time
                // since the script started loading, however fast the width
                // itself stabilized.
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

    tabs = st.tabs(["Methodology", "Terminology", "Comparison", "Architecture Deep Dive", "Query Explorer", "Chunking Experiment", "Embedding Quality"])

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

    # ---- Terminology ----------------------------------------------------
    with tabs[1]:
        st.header("Terminology")
        st.caption("Definitions for the retrieval and architecture concepts used throughout this dashboard. Metric definitions live in the Comparison tab, next to the tables they describe.")
        for term, definition in CONCEPT_TERMINOLOGY.items():
            with st.expander(term):
                st.markdown(definition)

    # ---- Comparison ------------------------------------------------------
    with tabs[2]:
        st.header("Cross-Architecture Comparison")

        st.subheader("In short")
        st.markdown(
            "**Query Decomposition RAG won overall** (0.675 F1) despite having the *worst* MRR "
            "(0.268) and second-worst Recall@k (0.517) of any architecture, a corpus-redundancy "
            "effect rather than better retrieval: it searches per sub-question and ends up with "
            "more distinct chunks per query (7.1 vs. 5.0 for the single-pass architectures), so "
            "it has more chances to surface the right named entity even when it misses the one "
            "designated 'gold' evidence sentence.\n\n"
            "**Hybrid RAG had the best retrieval quality** by every retrieval metric (0.650 "
            "Recall@k, 0.468 MRR), confirming that fusing dense and BM25 search recovers evidence "
            "neither finds reliably alone, but that didn't translate into the top F1 score: "
            "'best retrieval' and 'best final answer' turned out to be different questions here.\n\n"
            "**Corrective RAG's grading step didn't pay for itself.** Its correct-abstention rate "
            "on null queries (100%) is identical to Naive, Hybrid, and Reranked RAG's, all of "
            "which get there with no grading step at all, just the shared prompt's plain "
            "instruction not to guess. What the grading step *did* add is the worst "
            "incorrect-abstention rate in the experiment (40%, refusing answerable questions), at "
            "roughly double the LLM calls and wall-clock time per query.\n\n"
            "**The per-question-type breakdown tells a different story than the overall averages "
            "do.** Every architecture except HyDE ties near 0.92 F1 on Inference questions, but "
            "Comparison F1 spans 0.400 to 0.650 and Temporal F1 sits stuck at 0.250-0.450 for "
            "every architecture, a ceiling that looks like the model's own reasoning limits, not "
            "the retrieval mechanism.\n\n"
            "See the README's *What we learned* section for the full numbered findings, the "
            "mechanisms behind them, and what this task does and doesn't test."
        )

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
    with tabs[3]:
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

        st.markdown("### Evaluation metrics, defined")
        st.caption("What each column in the table above (and the Comparison tab's tables) actually measures.")
        for term, definition in RAG_METRIC_TERMINOLOGY.items():
            with st.expander(term):
                st.markdown(definition)

    # ---- Query Explorer ----------------------------------------------
    with tabs[4]:
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
    with tabs[5]:
        st.header("Chunking Sub-Experiment")
        if not CHUNKING_RESULTS_PATH.exists():
            st.warning("No chunking results yet. Run `python3 run_chunking_experiment.py` first.")
        else:
            strategies_meta, chunking_records = load_chunking_results()

            st.subheader("Setup")
            st.markdown(
                "Architecture held fixed at **Naive RAG** (no reranker or fusion step able to "
                "compensate for a weak first-pass chunk), varying only how the corpus was cut "
                "before indexing. Naive RAG re-indexes the entire 609-article corpus once per "
                f"strategy, then answers the same **{chunking_records['question_id'].nunique()} "
                "sampled queries** (8 per question type) against that strategy's index. This "
                "isolates what chunking alone does to retrieval and answer quality, holding the "
                "architecture, the model, and the query set fixed and changing only the chunk "
                "boundaries themselves."
            )

            st.subheader("What each strategy does, and why it's here")
            for key, info in CHUNKING_STRATEGY_INFO.items():
                if key not in strategies_meta:
                    continue
                with st.expander(info["label"]):
                    st.markdown(f"**What it does.** {info['what_it_does']}")
                    st.markdown(f"**Why it matters.** {info['why_it_matters']}")

            st.subheader("What we learned")
            st.markdown(
                "**Bigger fixed-size chunks retrieved the right evidence more often, but that "
                "didn't translate into the best answers.** Fixed 512 has the best Recall@k (0.750) "
                "by a clear margin, a bigger chunk is simply less likely to miss a fact entirely, "
                "but its F1 (0.562) lands in the middle of the pack, behind strategies with lower "
                "or equal recall. More surrounding, possibly irrelevant text rides along with the "
                "fact that matters.\n\n"
                "**Sentence-aware chunking beat both smaller and larger fixed-size cuts on the "
                "metric it should most directly affect.** Never splitting a fact-bearing sentence "
                "across a boundary gave Sentence chunking the second-best Recall@k (0.708) using "
                "much smaller chunks (155 tokens average) than Fixed 512 (460 tokens).\n\n"
                "**Semantic chunking tied for the best F1 despite chunk sizes varying wildly**, "
                "from 1 token to 608. The smallest chunks are likely isolated short sentences that "
                "got cut off on their own, which is probably also why Semantic chunking has the "
                "lowest faithfulness (0.632) and the lowest correct-abstention rate (87.5%) in this "
                "sub-experiment: a too-short chunk can still register as topically close enough to "
                "retrieve, without carrying enough context to ground an answer well.\n\n"
                "32 queries (8 per type) is small enough that a single query flipping from correct "
                "to incorrect moves a subgroup's F1 by 0.125, so read the exact ranking above as "
                "directional, not precise. See the README's chunking section for the full numbers "
                "and the general chunking-practice takeaways this points to."
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

            st.subheader("Evaluation metrics, defined")
            st.caption(
                "What each column in the tables above actually measures. Chunking only changes "
                "what a retrieval call finds, not how many calls a query makes, so the operational "
                "metrics (tokens, latency, LLM calls) shown for architectures elsewhere in this "
                "dashboard aren't tracked here — see the Architecture Deep Dive tab for those."
            )
            for term, definition in CHUNKING_METRIC_TERMINOLOGY.items():
                with st.expander(term):
                    st.markdown(definition)

    # ---- Embedding Quality -----------------------------------------------
    with tabs[6]:
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
