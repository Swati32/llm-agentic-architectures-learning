# Project: LLM & Agentic Architectures Learning

Mini learning apps that explore LLM and agentic architecture techniques using Hugging Face — datasets from the `datasets` library, apps built in Streamlit. Every app's purpose is to compare techniques side by side, not just demo one.

## Git workflow

Commit straight to `main`. This is a personal learning repo with a single contributor, so skip the usual "branch first, then merge" step and commit directly, overriding the general default of branching off the repo's default branch before committing.

## Starting a new experiment

Before writing any code, discuss and agree on:

1. **Goal** — what question this experiment answers
2. **Experiment design** — what techniques/methods are being compared
3. **Dataset** — which Hugging Face dataset, and why it fits
4. **Hoped outcome** — what we expect to learn or confirm
5. **Latest research** — relevant papers/techniques worth grounding the experiment in

Only move to implementation once these are settled.

## Flagging notable findings

If a result during an experiment looks paper worthy, say so in chat as soon as you notice it. Don't just fold it quietly into the README's learnings section and move on.

Before flagging, check whether the effect is already established elsewhere, in a different field, under a different name, not just in LLM prompting research specifically. A result that has a clean, already-published explanation (even from an unrelated field, like classical ML) is a confirmation, not a finding. Say that plainly rather than calling it paper worthy. Flag something only when existing research doesn't already predict or explain it.

## Repo structure

```
experiments/
  <experiment-name>/
    app.py                        # Streamlit dashboard, reads results/ only, no live model calls
    run_experiment.py             # orchestrates: load data, run every technique, write results/
    techniques/                   # one module per technique/method being compared
    llm_client.py                 # the only file that calls the model (see Model backend below)
    data.py                       # dataset loading (via `datasets` library)
    metrics.py                    # evaluation + operational metric collection
    results/records.json          # committed: the dashboard needs it to work when deployed
    requirements.txt              # dashboard-only dependencies (lean)
    requirements-experiment.txt   # additional dependencies needed to rerun the experiment
    README.md                     # blog-style writeup (see below)
```

Each experiment is self-contained: the dashboard reads a committed results file and never calls a model itself, so it can be deployed on its own without needing whatever backend generated the results.

## Evaluation metrics

Choose metrics deliberately per experiment — don't log everything by default. State in the experiment's README *why* each chosen metric fits the technique being evaluated.

Choosing metrics up front isn't enough on its own. Once an experiment has run, add a "why these metrics, and which ones actually mattered" analysis: which of the chosen metrics turned out to be decisive in explaining the results, which were secondary context, and which didn't differentiate anything at all (worth saying plainly, not omitting). Ground this in the real numbers, not the metric's definition. This is standing practice for every experiment from here on: it belongs in the README (near Results) and in the dashboard's comparison view, not just the upfront metric list in Methodology. A metric list defined before the run tells a reader what you planned to look at; this tells them what actually explained what they're seeing.

**Operational metrics** (dashboard on the website, for any LLM call):
- Tokens used
- Time to first token
- Latency
- Error rate
- Context payload size
- Semantic cache hit rate
- Empty retrieval rate

**Agentic metrics** (for agent/multi-step experiments):
- Step / loop count
- Tool execution latency
- Tool error / retry rate
- Inter-agent handoff count
- Early termination rate
- State overhead

## Technique documentation (e.g. prompt engineering/optimization)

For every technique compared, the website must show:
- **Technique name**
- **Prompt/method used** (verbatim)
- **Result**
- **Comparison** against the other techniques in the same experiment (table or chart)

## Code style

- Minimal comments, minimal logging — code should read clearly without them
- Variable and function names should document intent (e.g. `retrieved_chunks_before_rerank`, not `chunks2`)
- No docstrings or type annotations on code that isn't being actively changed (matches global preference)

## README format (per experiment, blog-style)

Each experiment README should read like a short blog post:
- What we set up (the app, the techniques compared)
- What we compared and how (dataset, methods, metrics)
- What we learned (results, surprises, caveats)
- References to the papers/research that motivated the experiment
- A Terminology section defining the key technical terms the experiment leans on, in plain language, with enough detail (not just a one-line definition) that a reader unfamiliar with the concepts can follow the rest of the README. Use a concrete example or analogy where it helps. Standing practice for every experiment from here on, not just ones about training internals.

## README writing guidelines

These apply to every README.md in this repo.

* Write in simple, easy to read language. No fluff, no filler sentences.
* Never use a dash ("-" or "—") as punctuation inside a sentence. Rewrite the sentence instead, or split it into two sentences. (Hyphens inside a single word, like "zero-shot" or "coarse-to-fine", are fine.)
* Every paper or external reference must be a clickable link, not just a name and year.
* Mermaid diagrams and images are welcome wherever a picture explains the setup or a result faster than text. Keep them simple.
* Every finding in "What we learned" needs a theorized *why*, not just a description of what happened. Ground the theory in a cited paper or a mechanism (how the model or technique actually works), not speculation stated as fact. "Retrieval won by 20 points" is not a finished finding; "retrieval won by 20 points, likely because X, per [paper]" is.
* When a feature is removed or changed, grep every README (and the portfolio blog post, if one exists for that experiment) for references to it in the same change. Stale mentions of removed features are worse than no mention.
* Standing practice: whenever an experiment's README changes in a way that affects its findings, conclusions, or citations, in the same turn update the corresponding post in the `portfolio-website` repo's blog too (condensed for blog pacing, same citations). Don't treat this as optional or a later cleanup step. If no post exists yet for that experiment, say so and ask whether to create one, rather than silently skipping it.
* Explain technical concepts and techniques the way a teacher would to someone new to the domain, not the way a paper would to a peer. Define a term in plain language the moment it's introduced, ideally with a concrete analogy, rather than assuming it's already known just because it's standard ML vocabulary (gradient, quantization, rank, optimizer state, and the like all need this treatment). This applies everywhere an experiment explains itself, not just its README: a dashboard's technique write-ups and the portfolio blog post get the same plain-language treatment, not a denser one.

## Model backend

* Keep every call to the model behind one small module (`llm_client.py`: one function in, one typed result out, carrying content plus the operational metrics). No technique file should talk to an API or local server directly. This is what makes it possible to swap backends without touching a single technique's code.
* Hosted free tiers run out mid-experiment. Have a free local fallback in mind from the start: a small instruction-tuned model via [Ollama](https://ollama.com) works well and removes cost as a variable entirely, at the cost of running on local compute instead of dedicated inference hardware.
* Prefer a model's non-streaming response over hand-rolled streaming when running locally through Ollama: its own reported timing fields (`prompt_eval_duration`, `total_duration`, etc.) are more accurate than timing an HTTP round trip yourself, and avoids a real bug we hit where a naive streaming read can hang forever on a finished response.
* When a local server reuses its KV cache across calls that share a long system prompt (Ollama does this by default), Time to First Token will look artificially small relative to total latency. That's a property of local single-user serving, not the technique; say so in the README rather than let the number stand unexplained.
* Run long experiment scripts with `python3 -u` (or set `PYTHONUNBUFFERED=1`) when logging progress to a file in the background. Buffered stdout means none of your progress prints show up until the process exits, which looks exactly like a hang.

## Dashboard practices

* Every dashboard's header banner uses the same deep purple to pink gradient: `linear-gradient(135deg, #5B21B6, #DB2777)`, white title text, `padding:1.5rem 1.5rem`, `border-radius:0.5rem`, `margin-bottom:1.5rem`. This is the site's visual identity across experiments, not a per-experiment choice: never pick a different gradient or color scheme for a new dashboard's header.
* If a prompt has a large, repeated boilerplate section (a full label catalog, a long fixed instruction), collapse it in the display down to a one-line placeholder. Show what's actually distinctive about that call in full (the query, the technique's own instructions, any dynamically chosen examples).
* When comparing N techniques' prompts, show all N at once (e.g. one expander per technique, same query for all), not a single dropdown that only reveals one at a time. The point of the dashboard is comparison; make that comparison the default view, not something the reader has to click into N times to reconstruct.
* Don't duplicate the same content across tabs (a technique's description, its metrics) just to fill out a section. If two sections show the same thing two different ways, cut the weaker one.
* Keep the dashboard's own `requirements.txt` limited to what `app.py` actually imports. If running the experiment needs heavier packages (`datasets`, `sentence-transformers`, a GPU-friendly torch build) that the dashboard itself never touches, put those in a separate `requirements-experiment.txt`. This keeps the deployed dashboard fast to build and free to host.

## Tooling

- Hugging Face `datasets` library for all data. Some dataset repos still ship a deprecated Python loading script that current `datasets` versions refuse to run; check for a `legacy-datasets/` or otherwise Parquet-native mirror of the same data first.
- Streamlit for the app UI. Streamlit's own dedicated Spaces SDK is deprecated on Hugging Face; Spaces now needs the Docker SDK for a Streamlit app, and Docker Spaces require a paid PRO plan even on free CPU hardware. Deploy the dashboard to [Streamlit Community Cloud](https://share.streamlit.io) instead, which is free and native, straight from the GitHub repo.
- `huggingface_hub` CLI (`hf` / `huggingface-cli`) for auth and Hub operations.
