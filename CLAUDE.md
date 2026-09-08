# Project: LLM & Agentic Architectures Learning

Mini learning apps that explore LLM and agentic architecture techniques using Hugging Face — datasets from the `datasets` library, apps built in Streamlit, deployable to Hugging Face Spaces. Every app's purpose is to compare techniques side by side, not just demo one.

## Starting a new experiment

Before writing any code, discuss and agree on:

1. **Goal** — what question this experiment answers
2. **Experiment design** — what techniques/methods are being compared
3. **Dataset** — which Hugging Face dataset, and why it fits
4. **Hoped outcome** — what we expect to learn or confirm
5. **Latest research** — relevant papers/techniques worth grounding the experiment in

Only move to implementation once these are settled.

## Repo structure

```
experiments/
  <experiment-name>/
    app.py              # Streamlit app for this experiment
    techniques/         # one module per technique/method being compared
    data.py             # dataset loading (via `datasets` library)
    metrics.py          # evaluation + operational metric collection
    README.md           # blog-style writeup (see below)
```

Each experiment is self-contained and independently runnable/deployable as its own Streamlit app.

## Evaluation metrics

Choose metrics deliberately per experiment — don't log everything by default. State in the experiment's README *why* each chosen metric fits the technique being evaluated.

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

## Tooling

- Hugging Face `datasets` library for all data
- Streamlit for the app UI, deployable to Hugging Face Spaces
- `huggingface_hub` CLI (`hf` / `huggingface-cli`) for auth and Hub operations
