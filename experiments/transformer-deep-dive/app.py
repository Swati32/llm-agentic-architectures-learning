"""Streamlit dashboard: a from-scratch walkthrough of "Attention Is All
You Need" (Vaswani et al., 2017), a worked attention example computed by
run_experiment.py, and a tour of how BERT, GPT, T5, and the efficiency
and Mixture-of-Experts lines that followed changed the original design
and why. No live model calls happen here; the worked example is read from
results/records.json, computed once by run_experiment.py.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import diagrams

RESULTS_PATH = Path(__file__).parent / "results" / "records.json"

TERMINOLOGY = {
    "Token": (
        "A chunk of text the model treats as one unit, usually a word or "
        "part of a word. \"cats\" might be one token; a rarer word might get "
        "split into two or three. Everything the model reads or writes is a "
        "sequence of tokens, never raw text."
    ),
    "Embedding": (
        "A list of numbers (a vector) that stands in for a token. Every "
        "token has its own embedding, learned during training, and tokens "
        "with similar meanings end up with similar-looking embeddings. This "
        "is how the model turns words into something arithmetic can act on."
    ),
    "Dimension (as in \"d_model\")": (
        "How many numbers are in each embedding. A model with d_model=4096 "
        "represents every token as a list of 4096 numbers. More dimensions "
        "give the model more room to represent finer shades of meaning, at "
        "the cost of more computation per token."
    ),
    "Positional encoding": (
        "A second set of numbers, added to each token's embedding, that "
        "encodes where in the sentence that token sits. Without it, the "
        "model would see \"dog bites man\" and \"man bites dog\" as the exact "
        "same bag of tokens, since attention on its own has no notion of "
        "order. The original paper used fixed sine and cosine waves "
        "(Section 3.5); later designs like RoPE changed how position "
        "reaches the model without changing why it's needed."
    ),
    "Query, Key, Value (Q, K, V)": (
        "Three different projections of the same token embedding, each "
        "learned separately. A useful mental model: Query is what this "
        "token is looking for, Key is what each token (including itself) "
        "has to offer, and Value is the actual content that gets passed "
        "along once a match is found. Attention compares one token's Query "
        "against every token's Key to decide how much of each token's "
        "Value to mix in."
    ),
    "Dot product": (
        "Multiply two vectors' matching numbers together and add up the "
        "results. A single number comes out. When two vectors point in a "
        "similar direction, their dot product is large; when they point in "
        "unrelated directions, it's small. This is the basic similarity "
        "measure attention runs between a Query and every Key."
    ),
    "Softmax": (
        "Turns a list of raw numbers into a list of probabilities: every "
        "value becomes positive, and the whole list sums to exactly 1. "
        "Attention runs softmax over one token's raw similarity scores "
        "against every other token, turning \"how similar\" into \"what "
        "fraction of attention to pay,\" so the weighted mix that follows "
        "is a proper weighted average, not just raw scores."
    ),
    "Attention weight": (
        "The output of that softmax: one number per pair of tokens, saying "
        "how much of one token's Value gets mixed into another token's "
        "output. Weights for a single token's row always add up to 1, so "
        "attention is literally deciding how to split 100% of its focus "
        "across every token it can see."
    ),
    "Attention head": (
        "One full, independent run of the Query/Key/Value mechanism, using "
        "its own slice of the embedding. A model doesn't run attention "
        "once per layer, it runs it h times in parallel, each head free to "
        "pick up a different kind of relationship (one head might track "
        "grammar, another might track which pronoun refers to which noun). "
        "Splitting into more heads doesn't add parameters; it divides the "
        "same d_model across more, smaller attention computations."
    ),
    "Self-attention vs. cross-attention": (
        "Self-attention: a sequence attending to itself, every token's "
        "Query compared against Keys from the same sequence. This is what "
        "both the encoder and decoder do internally. Cross-attention: one "
        "sequence's Queries compared against a different sequence's Keys "
        "and Values, which is how the decoder reads what the encoder "
        "produced. A decoder-only model like GPT has no encoder to read, "
        "so it has no cross-attention at all, only self-attention."
    ),
    "Causal (masked) attention": (
        "Self-attention with one restriction added: a token's Query is "
        "only allowed to compare against Keys from itself and earlier "
        "tokens, never later ones. This is what makes a model like GPT "
        "generate text one token at a time without ever \"peeking\" at the "
        "answer it hasn't produced yet."
    ),
    "Residual connection": (
        "A shortcut that adds a layer's input directly to its output, "
        "instead of forcing every layer's output to be built from scratch. "
        "This makes very deep stacks of layers much easier to train, since "
        "a gradient always has a direct path backward through the shortcut, "
        "even if the layer's own computation contributes little "
        "([He et al., 2015](https://arxiv.org/abs/1512.03385), the residual "
        "idea the transformer paper adopted)."
    ),
    "Layer normalization": (
        "Rescales the numbers coming out of a layer so they sit in a "
        "consistent, well-behaved range before the next layer sees them. "
        "Without it, values can drift to extremes as they pass through "
        "many stacked layers, making training unstable."
    ),
    "Feed-forward network / MLP": (
        "A small two-layer network applied to each token's vector, "
        "independently and identically, after attention has mixed "
        "information between tokens. If attention is where tokens talk to "
        "each other, the feed-forward layer is where each token thinks on "
        "its own about what it just heard."
    ),
    "Encoder / Decoder": (
        "The encoder reads an entire input at once, with every token "
        "allowed to attend to every other token (bidirectional). The "
        "decoder generates output one token at a time, restricted to "
        "causal attention over what it has produced so far, plus (in the "
        "original architecture) cross-attention back into the encoder's "
        "output. A model can use both (T5), only the encoder (BERT), or "
        "only the decoder (GPT)."
    ),
    "Autoregressive generation": (
        "Producing output one token at a time, feeding each generated "
        "token back in as part of the input for generating the next one. "
        "This is how every GPT-style model writes text: predict one token, "
        "append it, predict the next, repeat."
    ),
    "Logits": (
        "The model's raw, unnormalized scores over its entire vocabulary "
        "for \"what token comes next,\" before softmax turns them into "
        "probabilities that sum to 1. The final step of the whole "
        "architecture is exactly this: a linear layer followed by softmax, "
        "producing a probability for every possible next token."
    ),
    "KV cache": (
        "During generation, the Key and Value vectors for every token "
        "already produced don't need to be recomputed at each new step, "
        "so the model stores them. That storage is the KV cache, and it "
        "grows with every token generated. It's a major memory cost for "
        "long generations, which is exactly what grouped-query attention "
        "and multi-head latent attention (see Evolution) were built to cut "
        "down."
    ),
    "Sparse vs. dense model": (
        "A dense model runs every parameter for every token. A sparse "
        "model, like a Mixture-of-Experts model, has far more total "
        "parameters than it actually uses per token, since a router picks "
        "only a handful of \"expert\" sub-networks to run for each one. "
        "More total capacity, without paying its full compute cost on "
        "every single token."
    ),
}

FAMILY_ORDER = ["Original", "Encoder-only", "Decoder-only", "Encoder-decoder", "Efficiency", "Sparse (MoE)"]


@st.cache_data
def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def matrix_df(matrix, row_labels, col_prefix="d"):
    cols = [f"{col_prefix}{i}" for i in range(len(matrix[0]))]
    return pd.DataFrame(matrix, index=row_labels, columns=cols)


def _weight_shade(value: float) -> str:
    """Purple-intensity background for an attention weight in [0, 1], no
    matplotlib needed, to keep this dashboard's requirements.txt lean."""
    alpha = max(0.0, min(1.0, float(value)))
    text_color = "#FFFFFF" if alpha > 0.55 else "#1F2937"
    return f"background-color: rgba(91, 33, 182, {alpha:.2f}); color: {text_color}"


st.set_page_config(page_title="Attention Is All You Need: A Deep Dive", layout="wide")

st.markdown(
    """
    <div style="background:linear-gradient(135deg, #5B21B6, #DB2777);padding:1.5rem 1.5rem;border-radius:0.5rem;margin-bottom:1.5rem;">
        <h1 style="color:white;margin:0;font-size:2rem;">Attention Is All You Need: A Deep Dive</h1>
        <p style="color:#EDE9FE;margin:0.5rem 0 0 0;">How the original transformer works, traced through a real worked example, and how BERT, GPT, T5, and modern efficiency and Mixture-of-Experts designs changed it, and why.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not RESULTS_PATH.exists():
    st.warning(
        "No worked-example results yet. Run `python run_experiment.py` in "
        "this experiment's folder first. This dashboard reads from "
        "`results/records.json`."
    )
    st.stop()

data = load_results()

tab_walkthrough, tab_example, tab_evolution, tab_terms = st.tabs(
    ["Paper Walkthrough", "Worked Example", "Evolution Since 2017", "Terminology"]
)

# ---------------------------------------------------------------------------
with tab_walkthrough:
    st.header("Why this paper exists")
    st.markdown(
        "Before 2017, the best sequence models (translation, summarization) "
        "were built from recurrent neural networks (RNNs), which read a "
        "sentence one token at a time, in order, carrying a running summary "
        "forward as they went. That has two costs. First, it's slow to "
        "train: token 50 can't be processed until tokens 1 through 49 have "
        "been, so there's no way to process a sequence in parallel. Second, "
        "information from early tokens has to survive being carried through "
        "every step to reach a token much later in the sentence, and it "
        "tends to fade the farther it travels.\n\n"
        "[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) proposed "
        "removing recurrence entirely, and replacing it with attention: "
        "every token looks directly at every other token it needs, in one "
        "step, regardless of distance between them. That single change is "
        "the paper's title: attention, it turns out, is all you need, no "
        "recurrence, no convolution."
    )

    st.header("Tokenization, embeddings, and position")
    st.markdown(
        "The model never sees raw text. A sentence is first split into "
        "tokens, and every token is looked up in a table of embeddings, "
        "vectors the model learned during training. Two tokens with similar "
        "meaning end up with similar embeddings, which is what gives the "
        "model any notion of meaning to work with at all.\n\n"
        "But embeddings alone say nothing about order: attention will "
        "compare every token to every other token regardless of where they "
        "sit, so \"dog bites man\" and \"man bites dog\" would look identical "
        "without help. The paper's fix is a positional encoding: a second "
        "vector, one per position, built from sine and cosine waves at "
        "different frequencies, added directly onto the token embedding "
        "before anything else happens. That sum, token meaning plus "
        "position, is what actually enters the model."
    )
    st.markdown(diagrams.render("tokenization_embedding"), unsafe_allow_html=True)

    st.header("Scaled dot-product attention: the core mechanism")
    st.markdown(
        "This is the one idea the entire paper is built around. For every "
        "token, the model builds three vectors from its embedding, using "
        "three separate learned projections:\n\n"
        "- **Query (Q):** what this token is looking for\n"
        "- **Key (K):** what each token (including itself) has to offer\n"
        "- **Value (V):** the actual content passed along if a match is found\n\n"
        "To compute one token's output, its Query is compared against "
        "*every* token's Key using a dot product, a similarity score. Those "
        "scores are scaled down by dividing by the square root of the Key "
        "dimension (this keeps the scores from growing too large as "
        "dimension increases, which would push softmax into a region where "
        "its gradient is nearly flat and learning stalls), then passed "
        "through softmax so they become a proper set of weights that sum to "
        "1. Finally, those weights are used to build a weighted average of "
        "every token's Value. The formula the paper gives (Section 3.2.1) "
        "is exactly this:"
    )
    st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
    st.markdown(diagrams.render("scaled_dot_product_attention"), unsafe_allow_html=True)
    st.caption("See the Worked Example tab for this computed on a real 3-token sentence, with real numbers.")

    st.header("Multi-head attention")
    st.markdown(
        "Running attention once per layer would force every kind of "
        "relationship a token might need (grammar, reference, topic) to "
        "share one single similarity computation. Instead, the paper splits "
        "d_model into h smaller pieces and runs h independent attention "
        "computations in parallel, called heads, each with its own learned "
        "Q, K, V projections into that smaller space. Their outputs are "
        "concatenated back together and passed through one more learned "
        "projection (W_O). Splitting into more heads doesn't add "
        "parameters; each head just works in a smaller slice of the same "
        "total dimension, free to specialize on a different kind of pattern."
    )
    st.markdown(diagrams.render("multi_head_attention"), unsafe_allow_html=True)

    st.header("Around attention: residuals, normalization, feed-forward")
    st.markdown(
        "Attention alone isn't the whole layer. Around it, every block adds "
        "three more pieces, each solving a specific problem:\n\n"
        "- **Residual connections:** each sub-layer's output is added back "
        "to its own input, rather than replacing it outright. This gives "
        "gradients a direct path backward through a deep stack of layers, "
        "which is what makes stacking many layers (the paper uses 6 of "
        "each) trainable at all ([He et al., 2015](https://arxiv.org/abs/1512.03385)).\n"
        "- **Layer normalization:** rescales the numbers flowing between "
        "layers so they stay in a stable, consistent range, preventing "
        "values from drifting to extremes as they pass through many layers.\n"
        "- **Feed-forward network (MLP):** after attention has let tokens "
        "exchange information, a small two-layer network is applied to "
        "each token's vector independently. If attention is tokens talking "
        "to each other, this is each token thinking alone about what it "
        "just heard."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("One encoder block")
        st.markdown(diagrams.render("encoder_block"), unsafe_allow_html=True)
    with col2:
        st.subheader("One decoder block")
        st.markdown(diagrams.render("decoder_block"), unsafe_allow_html=True)

    st.header("The full architecture")
    st.markdown(
        "Stack six of each block, wire the decoder's cross-attention to "
        "read the encoder's final output, and add one last linear layer "
        "plus softmax over the vocabulary to turn the decoder's output into "
        "next-token probabilities. This is the complete architecture from "
        "the paper's Figure 1, used originally for machine translation: "
        "the encoder reads the full source sentence at once, and the "
        "decoder generates the translation one token at a time, causally, "
        "cross-attending back into the encoder at every step."
    )
    st.markdown(diagrams.render("full_architecture"), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
with tab_example:
    st.header('Tracing "cats chase mice" through attention')
    st.markdown(
        "The numbers below are real, computed arithmetic (see "
        "[`run_experiment.py`](run_experiment.py)), not illustrations drawn "
        "by hand. To keep every number small enough to read in a table, the "
        "embeddings and projection weights are **hand-picked, not learned "
        "from data or a trained model**. That means the specific attention "
        "pattern below isn't claiming anything about what a real trained "
        "model would do with this sentence; it exists purely to make the "
        "mechanism from the previous tab concrete. `d_model = 4` here, "
        "against thousands in a real model, for the same reason."
    )

    sh = data["single_head"]
    tokens = sh["tokens"]

    st.subheader("1. Token embeddings (d_model = 4)")
    st.dataframe(matrix_df(sh["embeddings"], tokens))

    st.subheader("2. Project to Query, Key, Value")
    st.markdown("Each embedding is multiplied by its own learned weight matrix (`W_Q`, `W_K`, `W_V`).")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Q**")
    col1.dataframe(matrix_df(sh["Q"], tokens))
    col2.markdown("**K**")
    col2.dataframe(matrix_df(sh["K"], tokens))
    col3.markdown("**V**")
    col3.dataframe(matrix_df(sh["V"], tokens))

    st.subheader("3. Raw scores -> scaled -> softmax -> attention weights")
    st.markdown(
        f"Raw scores are `Q @ K^T`. Scaled scores divide by `sqrt(d_k) = "
        f"sqrt({sh['d_k']}) = {sh['d_k'] ** 0.5:.2f}`. Softmax turns each row into weights that sum to 1."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scaled scores**")
        st.dataframe(matrix_df(sh["scaled_scores"], tokens, col_prefix="to "))
    with col2:
        st.markdown("**Attention weights (after softmax)**")
        weights_df = matrix_df(sh["weights"], tokens, col_prefix="to ")
        st.dataframe(weights_df.style.format("{:.3f}").map(_weight_shade))
    st.caption(
        "Read row 'chase': it splits its attention almost evenly between 'cats' and 'mice', "
        "its subject and object, a tidy coincidence of these hand-picked numbers, worth noticing but not reading too much into."
    )

    st.subheader("4. Weighted sum of Values -> output")
    st.markdown(
        "Each token's output row is its attention-weight row multiplied "
        "against every token's Value vector and summed, a weighted average, "
        "not a hard pick of one token's Value."
    )
    st.dataframe(matrix_df(sh["output"], tokens))

    st.divider()
    st.header("Same example, split across 2 heads")
    mh = data["multi_head"]
    st.markdown(
        f"`d_model = 4` is split into `{mh['n_heads']}` heads of "
        f"`d_head = {mh['d_head']}` each. Every head repeats the exact same "
        "four steps above, independently, in its own smaller subspace."
    )
    head_cols = st.columns(mh["n_heads"])
    for i, head in enumerate(mh["heads"]):
        with head_cols[i]:
            st.markdown(f"**Head {head['head'] + 1}**")
            st.markdown("Attention weights")
            st.dataframe(
                matrix_df(head["weights"], tokens, col_prefix="to ").style.format("{:.3f}").map(_weight_shade)
            )
            st.markdown("Output")
            st.dataframe(matrix_df(head["output"], tokens))

    st.markdown("**Concatenate both heads' outputs, then project through `W_O`:**")
    st.dataframe(matrix_df(mh["output"], tokens))
    st.caption(
        "W_O is the identity matrix here, purely to keep the final numbers "
        "matching the concatenation exactly and easy to trace by eye; a "
        "real model learns a non-trivial W_O."
    )

# ---------------------------------------------------------------------------
with tab_evolution:
    st.markdown(
        "The 2017 architecture is a full encoder-decoder built for "
        "translation. Almost nothing since has abandoned attention itself; "
        "instead, each design below kept the mechanism and changed either "
        "which half of the architecture it uses, or how attention is "
        "computed to make it cheaper to run at scale. Each entry names the "
        "motivation first, since that's what actually explains the design."
    )

    with st.expander("BERT (2018) — encoder-only, for understanding text", expanded=True):
        st.markdown(diagrams.render("bert"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** Many real tasks (classification, sentiment, "
            "answering a question about a passage) don't need to *generate* "
            "text at all; they need to deeply *understand* it. "
            "[Devlin et al., 2018](https://arxiv.org/abs/1810.04805) kept "
            "only the encoder half of the original architecture, where "
            "every token can attend to every other token in both "
            "directions, and trained it by hiding random words and asking "
            "the model to predict them (masked language modeling), which "
            "forces the model to use both left and right context at once. "
            "**Trade-off:** BERT has no decoder and no causal mask, so it "
            "cannot generate text one token at a time the way GPT can."
        )

    with st.expander("GPT (2018–) — decoder-only, for generating text", expanded=True):
        st.markdown(diagrams.render("gpt"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** If the goal is generating fluent text, "
            "bidirectional attention is actually a liability: a model that "
            "could see the whole sentence at once, including the word it's "
            "supposed to predict, would have nothing left to learn. "
            "[Radford et al., 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) "
            "kept only the decoder half, with causal (masked) self-attention "
            "and no cross-attention at all, since there's no separate "
            "encoder to read from. Trained simply to predict the next "
            "token, over and over, on huge amounts of text. This is the "
            "architecture behind every modern chat-style LLM: GPT, Llama, "
            "Mistral, and (at the level of publicly known trends; see the "
            "caveat at the end of this tab) Claude are all decoder-only "
            "descendants of this half of the original design."
        )

    with st.expander("T5 (2019) — encoder-decoder, everything as text-to-text", expanded=True):
        st.markdown(diagrams.render("t5"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** [Raffel et al., 2019](https://arxiv.org/abs/1910.10683) "
            "kept the *full* original architecture, encoder and decoder "
            "both, but changed how every task is framed: translation, "
            "summarization, classification, even regression are all "
            "rewritten as \"input text in, output text out,\" using a task "
            "prefix in the input (like `\"summarize: ...\"`). This doesn't "
            "change the mechanism at all; it's a framing choice that lets "
            "one architecture and one training recipe handle wildly "
            "different tasks without task-specific output layers."
        )

    st.subheader("The efficiency line: making attention cheaper to run")
    st.markdown(
        "The next set of changes isn't about which half of the "
        "architecture to keep. It's about a cost that becomes serious once "
        "models generate long sequences: at every generation step, a model "
        "needs the Key and Value vectors of every token generated so far "
        "(the KV cache), and computing full attention over a long sequence "
        "gets expensive in both memory and time."
    )

    with st.expander("RoPE — rotary position embeddings", expanded=False):
        st.markdown(diagrams.render("rope"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** The original sinusoidal positional encoding is "
            "added once, at the input, and has to somehow still be readable "
            "by attention many layers later. "
            "[Su et al., 2021](https://arxiv.org/abs/2104.09864) instead "
            "bakes position directly into the attention computation itself: "
            "each Query and Key vector is rotated by an angle proportional "
            "to its position, at every layer, so the dot product between "
            "two tokens' Query and Key naturally depends on their *relative* "
            "distance, not just their absolute positions. This also "
            "generalizes better to sequence lengths longer than anything "
            "seen during training, which is part of why nearly every modern "
            "open LLM (Llama, Mistral, DeepSeek, Qwen) uses it."
        )

    with st.expander("Grouped-query attention (GQA)", expanded=False):
        st.markdown(diagrams.render("gqa"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** In standard multi-head attention, every Query "
            "head has its own Key and Value head, so the KV cache grows "
            "with the full head count. Multi-query attention "
            "([Shazeer, 2019](https://arxiv.org/abs/1911.02150)) pushed to "
            "the opposite extreme: every Query head shares one single K/V "
            "head, shrinking the cache a lot but giving up some quality. "
            "[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245) "
            "proposed grouped-query attention as the middle ground: Query "
            "heads are split into a handful of groups, and each group "
            "shares one K/V head. It's the setting most current open models "
            "(Llama 2 and later, Mistral) actually ship with."
        )

    with st.expander("FlashAttention", expanded=False):
        st.markdown(diagrams.render("flashattention"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** This one changes nothing about the math from "
            "Section 3.2.1: same Q, K, V, same softmax, same output. What "
            "[Dao et al., 2022](https://arxiv.org/abs/2205.14135) noticed is "
            "that standard attention is bottlenecked by *moving* the full "
            "N x N score matrix between slow GPU memory (HBM) and fast "
            "on-chip memory (SRAM), not by the matrix multiplication "
            "itself. FlashAttention processes attention in small tiles that "
            "fit in fast memory and never materializes the full matrix at "
            "all, giving the exact same output much faster and with far "
            "less memory. It's an implementation-level optimization, not a "
            "new architecture, but it's a big part of why long-context "
            "models became practical."
        )

    st.subheader("Sparse models: Mixture of Experts (MoE)")
    with st.expander("Mixture of Experts", expanded=False):
        st.markdown(diagrams.render("moe"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** A dense model's every parameter runs on every "
            "token, so making the model bigger always makes every token "
            "more expensive to process. "
            "[Shazeer et al., 2017](https://arxiv.org/abs/1701.06538) "
            "proposed replacing the single feed-forward network in each "
            "block with many smaller feed-forward \"experts,\" plus a small "
            "learned router that picks only a few experts to actually run "
            "for each token. The model's *total* parameter count can grow "
            "far larger than a dense model's, while the compute spent per "
            "token stays close to a much smaller dense model's, since most "
            "experts sit idle for any given token. "
            "[Fedus et al., 2021](https://arxiv.org/abs/2101.03961)'s Switch "
            "Transformer showed this scales well even routing to just a "
            "single expert per token."
        )

    with st.expander("DeepSeek — multi-head latent attention + DeepSeekMoE", expanded=False):
        st.markdown(diagrams.render("deepseek"), unsafe_allow_html=True)
        st.markdown(
            "**Motivation.** DeepSeek's models combine two efficiency ideas "
            "from opposite ends of the architecture. Multi-head latent "
            "attention ([DeepSeek-AI, 2024, DeepSeek-V2](https://arxiv.org/abs/2405.04434)) "
            "compresses what would normally be full per-head Key and Value "
            "vectors into one small shared latent vector before caching it, "
            "decompressing on the fly when attention runs, which cuts KV "
            "cache size well beyond what grouped-query attention alone "
            "achieves. DeepSeekMoE "
            "([Dai et al., 2024](https://arxiv.org/abs/2401.06066)) reworks "
            "the expert side of MoE: instead of a few large experts, it "
            "uses many more, smaller, finer-grained routed experts for "
            "sharper specialization, plus a small number of shared experts "
            "that every token always runs through, to capture the common "
            "patterns every token needs without making every routed expert "
            "re-learn them. Both ideas attack the same underlying pressure "
            "as GQA and MoE above, just further along the same two axes: "
            "smaller cache per token, more specialization per parameter "
            "spent."
        )

    st.subheader("A note on Perplexity and Claude")
    st.markdown(
        "**Perplexity** isn't a transformer architecture at all; it's a "
        "product built on top of an existing LLM, combining it with web "
        "retrieval so answers can cite live sources. There's no separate "
        "architecture to diagram here, just an existing decoder-only model "
        "(from one of several providers, depending on the mode) wired up to "
        "a search layer, closer in spirit to retrieval-augmented generation "
        "than to a new architectural family.\n\n"
        "**Claude's** internal architecture isn't publicly published by "
        "Anthropic, no paper laying out layer counts, attention variant, or "
        "MoE structure, the way there is for GPT-2, T5, or DeepSeek. What "
        "can be said honestly, at the level of publicly known industry "
        "trends rather than confirmed internals, is that it's a "
        "decoder-only transformer descendant like the rest of this section, "
        "using some form of efficient attention to support long context. "
        "Presenting more detail than that as a real diagram would be "
        "inventing internals that haven't been published, so this dashboard "
        "doesn't."
    )

# ---------------------------------------------------------------------------
with tab_terms:
    st.header("Terminology")
    st.caption("Definitions for the terms used throughout this page, in the order they first come up.")
    for term, definition in TERMINOLOGY.items():
        with st.expander(term):
            st.markdown(definition)
