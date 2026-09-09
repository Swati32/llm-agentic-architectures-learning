"""Streamlit dashboard: methodology and terminology, the cross-technique
comparison table and charts, and a per-technique deep-dive into how each
one actually works. No live model calls or training runs happen here;
everything is read from results/records.json, computed by run_experiment.py.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"

TERMINOLOGY = {
    "Parameter / weight": (
        "A single number inside the model that it learned during training. A "
        "model with \"8 billion parameters\" has 8 billion of these numbers. "
        "Picture a huge mixing board with billions of dials; training is turning "
        "each dial until the model gives good answers."
    ),
    "Trainable parameter": (
        "A parameter this training run is allowed to change. Full fine-tuning "
        "changes all of them. PEFT changes only a small slice and leaves the rest "
        "untouched. This one number is why PEFT is cheaper: fewer trainable "
        "parameters means less memory, faster training, and a smaller file to "
        "save at the end."
    ),
    "Frozen weights": (
        "Parameters that are not being trained. The model still uses them to "
        "compute its answer, but nothing about them changes, so they cost far "
        "less memory than a trainable parameter does."
    ),
    "Gradient": (
        "A number that tells a trainable parameter which way to move, and by "
        "how much, to make the model's next answer a little better. Every "
        "trainable parameter gets its own gradient, recomputed after each "
        "attempt."
    ),
    "Optimizer / Adam": (
        "The rule that actually moves a parameter, using its gradient. Adam "
        "([Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)) is the most "
        "common choice for training language models today, because it gives "
        "each parameter its own step size based on that parameter's recent "
        "history, instead of using one fixed step size for everything."
    ),
    "Optimizer state": (
        "Extra numbers Adam keeps for every trainable parameter, to remember "
        "that parameter's recent history. In practice, this means each "
        "trainable parameter ends up costing about 4 times its own size in "
        "memory once its gradient and Adam's extra numbers are added in. That's "
        "the real reason PEFT saves memory: fewer trainable parameters means "
        "paying that 4x cost far fewer times, not just skipping a smaller update."
    ),
    "Rank (as in \"low-rank\")": (
        "In LoRA, the update to a weight is built from two small matrices "
        "instead of one big one. Rank is how big those small matrices are. A "
        "higher rank can express a more detailed update, but adds more "
        "trainable parameters too. The bet behind picking a low rank at all is "
        "that the update a model actually needs is simpler than its full size "
        "suggests ([Aghajanyan et al., 2020](https://arxiv.org/abs/2012.13255))."
    ),
    "Quantization (4-bit)": (
        "Storing a number in less space than usual, at some cost to precision. "
        "QLoRA stores the frozen part of the model this way to save memory, "
        "while keeping the small trainable part at normal precision, since "
        "that's the part that actually needs to be updated accurately."
    ),
    "Forward pass / backward pass": (
        "The forward pass is the model reading an input and producing an "
        "answer. The backward pass works backward from how wrong that answer "
        "was, to figure out each trainable parameter's gradient. Frozen "
        "parameters skip the backward pass entirely, which is part of why PEFT "
        "trains faster, not just lighter, than full fine-tuning."
    ),
    "Memory footprint (fine-tuning)": (
        "How much memory training actually needs: the weights, the gradients, "
        "Adam's optimizer state, and the activations kept from the forward "
        "pass. This experiment only compares the first three, since those are "
        "the ones that change depending on which technique you pick; the "
        "activations cost roughly the same no matter which technique is used."
    ),
}

FAMILY_ORDER = ["Baseline", "Reparameterization", "Bottleneck module", "Soft prompt", "Selective"]


@st.cache_data
def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def records_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records).set_index("technique")
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER, ordered=True)
    return df.sort_values(["family"])


st.set_page_config(page_title="PEFT / LoRA / QLoRA: Theoretical Comparison", layout="wide")

st.markdown(
    """
    <div style="background:linear-gradient(135deg, #6D28D9, #DB2777);padding:1.5rem 1.5rem;border-radius:0.5rem;margin-bottom:1.5rem;">
        <h1 style="color:white;margin:0;font-size:2rem;">PEFT / LoRA / QLoRA: Theoretical Comparison</h1>
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

data = load_results()
df = records_df(data["records"])
rank_sweep = pd.DataFrame(data["lora_rank_sweep"]).set_index(pd.Index([c["rank"] for c in [r["config"] for r in data["lora_rank_sweep"]]], name="rank"))

tab_methodology, tab_comparison, tab_deep_dive = st.tabs(
    ["Methodology & Terminology", "Comparison", "Technique Deep-Dive"]
)

with tab_methodology:
    st.header("Goal")
    st.markdown(
        "Build clear intuition for why parameter-efficient fine-tuning (PEFT) "
        "methods exist and how they differ mechanically from each other and from "
        "full fine-tuning, using real computed numbers rather than prose alone, "
        "without needing an actual training run."
    )

    st.header("Scenario")
    st.markdown(
        f"**Reference model:** `{data['reference_model']}` — the same model used in "
        "the [intent-classification-prompting](../intent-classification-prompting/README.md) "
        f"experiment, {data['total_params']:,} parameters.\n\n"
        f"**Scenario:** {data['scenario']}. This scenario is not actually run; it "
        "exists to give every technique's numbers a concrete, consistent backdrop "
        "instead of floating as abstract percentages."
    )

    st.header("Why no training run")
    st.markdown(
        "Every other experiment in this repo calls a model. This one computes "
        "trainable-parameter counts and memory footprints directly from the "
        "model's published architecture (hidden size, layer count, attention head "
        "counts), the same numbers a training run would produce, without needing "
        "GPU time this repo doesn't reliably have. `model_config.py` holds the "
        "architecture constants and the memory-footprint formula; each file in "
        "`techniques/` computes one technique's numbers from them."
    )

    st.header("Terminology")
    st.caption("Definitions for the terms used throughout this page.")
    for term, definition in TERMINOLOGY.items():
        with st.expander(term):
            st.markdown(definition)

with tab_comparison:
    st.header("How the five PEFT techniques compare")
    st.markdown(
        "All five PEFT techniques land in the same rough memory range, 3.8GB to "
        "15.4GB, next to full fine-tuning's 119.66GB. That's the headline: which "
        "PEFT technique you pick barely matters next to the decision to use PEFT "
        "at all. The differences between them show up in where the trainable "
        "parameters actually sit, not in how many there are:\n\n"
        "- **Cheapest overall: QLoRA (3.79GB).** Same trainable parameters as "
        "LoRA, quarter the memory, because it also compresses the frozen weights.\n"
        "- **Cheapest without touching precision: LoRA (15.00GB, 0.0424% "
        "trainable).** The reference point every other technique here is "
        "measured against.\n"
        "- **Most trainable parameters: Adapters (33.8M, roughly 10x LoRA's "
        "count),** and the only technique here that adds a permanent inference "
        "cost, since its modules can't be merged back into the frozen weights "
        "the way LoRA's can.\n"
        "- **Cheapest by parameter count that still touches weights: Prefix-"
        "tuning (5.24M),** but its real cost doesn't show up in this table: it "
        "spends context window positions instead of memory.\n"
        "- **Zero trainable parameters: BitFit,** not because it's the most "
        "efficient technique, but because `llama3.1:8b` has no bias terms for "
        "it to select. On a BERT-style model with biases in every layer, this "
        "row would look completely different.\n\n"
        "In short: LoRA and QLoRA win on raw memory because their update sits "
        "*alongside* the frozen weights and merges away after training. Adapters "
        "and Prefix-tuning both add something that stays around at inference "
        "time, in different forms (a module in Adapters' case, context space in "
        "Prefix-tuning's), which the parameter/memory numbers alone don't fully "
        "capture."
    )

    st.subheader("Trainable parameters and memory, all techniques")
    display_df = df[["family", "trainable_params", "trainable_pct", "frozen_params", "memory_gb"]].rename(
        columns={
            "family": "Family",
            "trainable_params": "Trainable Params",
            "trainable_pct": "Trainable %",
            "frozen_params": "Frozen Params",
            "memory_gb": "Memory to Fine-Tune (GB)",
        }
    )
    st.dataframe(
        display_df.style.format(
            {
                "Trainable Params": "{:,}",
                "Trainable %": "{:.4f}%",
                "Frozen Params": "{:,}",
                "Memory to Fine-Tune (GB)": "{:.2f}",
            }
        ),
        width="stretch",
    )

    st.caption(
        "BitFit trains 0 parameters on this model: Llama-family architectures have "
        "no bias terms in their linear layers, and RMSNorm has only a weight, no "
        "bias, so there's nothing for BitFit to select. See Terminology and the "
        "README for why."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trainable %")
        chart_df = df[["trainable_pct"]].copy()
        chart_df["trainable_pct"] = chart_df["trainable_pct"].clip(lower=0.05)
        st.bar_chart(chart_df, y_label="Trainable %", height=350)
        st.caption(
            "Full fine-tuning dwarfs every PEFT technique on a linear scale; see the "
            "table above for exact percentages. BitFit's bar is clipped to a visible "
            "sliver; its true value is 0%."
        )
    with col2:
        st.subheader("Memory to fine-tune (GB)")
        st.bar_chart(df[["memory_gb"]], height=350)
        st.caption("Full fine-tuning's bar is off this scale on purpose: see the table above for its exact value.")

    st.header("LoRA: trainable parameters vs. rank")
    st.markdown(
        "Rank sets how many independent directions LoRA's update can move in. "
        "Trainable parameter count scales linearly with it, holding everything "
        "else (which matrices are targeted, the base model) fixed."
    )
    st.line_chart(rank_sweep[["trainable_params"]], height=300)
    st.dataframe(
        rank_sweep[["trainable_params", "trainable_pct", "memory_gb"]]
        .rename(columns={"trainable_params": "Trainable Params", "trainable_pct": "Trainable %", "memory_gb": "Memory (GB)"})
        .style.format({"Trainable Params": "{:,}", "Trainable %": "{:.4f}%", "Memory (GB)": "{:.2f}"})
    )

with tab_deep_dive:
    st.caption("Same scenario for every technique: fine-tuning llama3.1:8b to summarize SAMSum dialogues.")
    for technique_name in df.index:
        record = df.loc[technique_name]
        with st.expander(f"{technique_name} — {record['family']}"):
            st.markdown(f"**How it works:** {record['mechanism']}")
            st.markdown(f"**Why it exists:** {record['why']}")
            st.markdown(f"**Trade-offs:** {record['tradeoffs']}")
            if record["config"]:
                st.markdown(f"**Config:** `{record['config']}`")
            cols = st.columns(3)
            cols[0].metric("Trainable Params", f"{record['trainable_params']:,}")
            cols[1].metric("Trainable %", f"{record['trainable_pct']:.4f}%")
            cols[2].metric("Memory to Fine-Tune", f"{record['memory_gb']:.2f} GB")
