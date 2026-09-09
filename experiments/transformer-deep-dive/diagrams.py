"""Hand-built inline SVG diagrams for the Streamlit app. No plotting
library needed, and no new dependency for the dashboard: every diagram is
a plain string of SVG markup, following the same approach used in
experiments/peft-comparison/diagrams.py.

One visual grammar runs through all of them: gray boxes are fixed
mechanism (nothing being learned right now), purple boxes are the
attention mechanism itself, blue boxes are the feed-forward/MLP path,
and amber boxes mark whatever is new or different about the design being
shown, so a reader can spot "what changed here" at a glance across the
evolution diagrams.
"""

GRAY = {"fill": "#E5E7EB", "stroke": "#9CA3AF", "text_color": "#374151"}
PURPLE = {"fill": "#5B21B6", "stroke": "#4C1D95", "text_color": "#FFFFFF"}
LIGHT_PURPLE = {"fill": "#DDD6FE", "stroke": "#7C3AED", "text_color": "#4C1D95"}
BLUE = {"fill": "#2563EB", "stroke": "#1D4ED8", "text_color": "#FFFFFF"}
AMBER = {"fill": "#D97706", "stroke": "#B45309", "text_color": "#FFFFFF"}
PINK = {"fill": "#DB2777", "stroke": "#9D174D", "text_color": "#FFFFFF"}

ARROW_COLOR = "#6B7280"


def _box(x, y, w, h, lines, fill, stroke, text_color, font_size=13, rx=10):
    cx = x + w / 2
    cy = y + h / 2 - (len(lines) - 1) * 8
    tspans = "".join(
        f'<tspan x="{cx}" dy="{0 if i == 0 else 16}">{line}</tspan>' for i, line in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2"/>'
        f'<text x="{cx}" y="{cy}" text-anchor="middle" font-size="{font_size}" '
        f'fill="{text_color}" font-family="sans-serif">{tspans}</text>'
    )


def _arrow(x1, y1, x2, y2, dashed=False, color=ARROW_COLOR, marker="arrow"):
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2" marker-end="url(#{marker})"{dash}/>'
    )


def _label(x, y, text, size=11, color=ARROW_COLOR, weight="normal"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
        f'fill="{color}" font-family="sans-serif" font-weight="{weight}">{text}</text>'
    )


def _wrap(inner, w, h, extra_defs=""):
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{w}px;display:block;margin:0.5rem auto;">'
        '<defs>'
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="{ARROW_COLOR}"/></marker>'
        '<marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="{PURPLE["stroke"]}"/></marker>'
        f'{extra_defs}'
        '</defs>'
        f"{inner}</svg>"
    )


# ---------------------------------------------------------------------------
# 1. Tokenization -> embedding -> positional encoding
# ---------------------------------------------------------------------------

def tokenization_embedding():
    w, h = 680, 220
    inner = (
        _label(70, 30, '"cats chase mice"', size=14, weight="bold", color="#374151")
        + _box(20, 50, 100, 40, ["cats"], **GRAY)
        + _box(130, 50, 100, 40, ["chase"], **GRAY)
        + _box(240, 50, 100, 40, ["mice"], **GRAY)
        + _label(180, 40, "tokens", size=11)
        + _arrow(70, 90, 70, 130)
        + _arrow(180, 90, 180, 130)
        + _arrow(290, 90, 290, 130)
        + _box(20, 130, 100, 40, ["[0.2, 0.8,", "...] "], **LIGHT_PURPLE, font_size=11)
        + _box(130, 130, 100, 40, ["[0.9, 0.1,", "...]"], **LIGHT_PURPLE, font_size=11)
        + _box(240, 130, 100, 40, ["[0.4, 0.5,", "...]"], **LIGHT_PURPLE, font_size=11)
        + _label(180, 120, "token embeddings (learned lookup)", size=11)
        + _label(420, 100, "+", size=24, weight="bold", color="#374151")
        + _box(460, 60, 190, 110, ["Positional encoding:", "sine/cosine waves,", "one per position,", "fixed (not learned)"], **AMBER, font_size=11)
        + _arrow(650, 115, 675, 115)
        + _label(180, 200, "Same token embedding at any position; the sum is what makes position visible to attention", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 2. Scaled dot-product attention (paper Figure 2, left)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention():
    w, h = 680, 260
    inner = (
        _box(20, 20, 90, 40, ["Q"], **PURPLE)
        + _box(20, 100, 90, 40, ["K"], **PURPLE)
        + _box(20, 180, 90, 40, ["V"], **PURPLE)
        + _arrow(110, 40, 200, 70)
        + _arrow(110, 120, 200, 90)
        + _box(200, 55, 140, 50, ["MatMul", "Q x K^T"], **GRAY, font_size=12)
        + _arrow(270, 105, 270, 130)
        + _box(200, 130, 140, 40, ["Scale", "/ sqrt(d_k)"], **GRAY, font_size=12)
        + _arrow(270, 170, 270, 195)
        + _box(200, 195, 140, 40, ["Softmax"], **PURPLE, font_size=12)
        + _arrow(340, 215, 420, 215)
        + _arrow(110, 200, 420, 240, dashed=False)
        + _box(420, 195, 140, 40, ["MatMul", "weights x V"], **GRAY, font_size=12)
        + _arrow(560, 215, 630, 215)
        + _label(655, 220, "Output", size=12)
        + _label(w / 2, 20, "Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) V", size=13, weight="bold", color="#374151")
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 3. Multi-head attention (paper Figure 2, right)
# ---------------------------------------------------------------------------

def multi_head_attention():
    w, h = 680, 260
    def head(y, label):
        return (
            _box(150, y, 90, 34, ["Linear"], **GRAY, font_size=11)
            + _box(260, y, 110, 34, [label], **PURPLE, font_size=11)
            + _arrow(240, y + 17, 260, y + 17)
        )
    inner = (
        _label(70, 30, "Q, K, V", size=12, weight="bold", color="#374151")
        + _arrow(40, 45, 40, 190)
        + _arrow(40, 65, 150, 65)
        + _arrow(40, 115, 150, 115)
        + _arrow(40, 165, 150, 165)
        + head(48, "head 1: attn")
        + head(98, "head 2: attn")
        + _label(300, 155, "...", size=16)
        + head(160, "head h: attn")
        + _arrow(370, 65, 450, 130)
        + _arrow(370, 115, 450, 135)
        + _arrow(370, 177, 450, 140)
        + _box(450, 110, 100, 60, ["Concat"], **GRAY, font_size=12)
        + _arrow(550, 140, 600, 140)
        + _box(600, 110, 60, 60, ["W_O"], **PURPLE, font_size=12)
        + _arrow(660, 140, 675, 140)
        + _label(w / 2, 20, "Each head attends in its own smaller subspace, then results are combined", size=12, color="#374151")
        + _label(w / 2, 235, "Splitting one attention into h heads costs nothing extra: d_model is divided across them, not multiplied", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 4/5. Encoder block and decoder block
# ---------------------------------------------------------------------------

def encoder_block():
    w, h = 420, 380
    inner = (
        _box(110, 340, 200, 30, ["Input embeddings + position"], **GRAY, font_size=11)
        + _arrow(210, 340, 210, 300)
        + _box(60, 250, 300, 50, ["Multi-Head Self-Attention", "(every token sees every token)"], **PURPLE, font_size=11)
        + _arrow(210, 250, 210, 215)
        + _label(340, 275, "+", size=18, weight="bold")
        + _arrow(370, 340, 370, 190, dashed=True)
        + _box(60, 165, 300, 50, ["Add & Norm", "(residual connection)"], **GRAY, font_size=11)
        + _arrow(210, 165, 210, 130)
        + _box(60, 80, 300, 50, ["Feed-Forward (MLP)", "same weights, every position"], **BLUE, font_size=11)
        + _arrow(210, 80, 210, 45)
        + _label(340, 105, "+", size=18, weight="bold")
        + _arrow(370, 250, 370, 65, dashed=True)
        + _box(60, 0, 300, 45, ["Add & Norm", "-> output (repeat Nx)"], **GRAY, font_size=11)
        + _label(w / 2, 375, "Stack this block N times (paper: N=6)", size=11)
    )
    return _wrap(inner, w, h)


def decoder_block():
    w, h = 460, 460
    inner = (
        _box(120, 415, 220, 30, ["Output embeddings, shifted right"], **GRAY, font_size=11)
        + _arrow(230, 415, 230, 375)
        + _box(60, 325, 340, 50, ["Masked Multi-Head Self-Attention", "(each token sees only earlier tokens)"], **PURPLE, font_size=11)
        + _arrow(230, 325, 230, 290)
        + _box(60, 240, 340, 45, ["Add & Norm"], **GRAY, font_size=11)
        + _arrow(230, 240, 230, 205)
        + _box(60, 155, 340, 50, ["Multi-Head Cross-Attention", "Q from decoder, K & V from encoder"], **PINK, font_size=11)
        + _arrow(230, 155, 230, 120)
        + _label(400, 130, "<- K, V", size=11, color=PINK["stroke"], weight="bold")
        + _box(60, 70, 340, 45, ["Add & Norm"], **GRAY, font_size=11)
        + _arrow(230, 70, 230, 45)
        + _box(60, 0, 340, 45, ["Feed-Forward (MLP) + Add & Norm", "-> output (repeat Nx)"], **BLUE, font_size=10)
        + _label(w / 2, 455, "Cross-attention is the only place the decoder reads the encoder's output", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 6. Full architecture
# ---------------------------------------------------------------------------

def full_architecture():
    w, h = 620, 300
    inner = (
        _label(140, 25, "Encoder stack (N x)", size=13, weight="bold", color="#374151")
        + _label(470, 25, "Decoder stack (N x)", size=13, weight="bold", color="#374151")
        + _box(40, 50, 200, 45, ["Self-Attention", "(bidirectional)"], **PURPLE, font_size=11)
        + _arrow(140, 95, 140, 120)
        + _box(40, 120, 200, 45, ["Feed-Forward"], **BLUE, font_size=11)
        + _label(140, 185, "...N times...", size=11)
        + _box(370, 50, 200, 45, ["Masked Self-Attention", "(causal)"], **PURPLE, font_size=11)
        + _arrow(470, 95, 470, 120)
        + _box(370, 120, 200, 45, ["Cross-Attention"], **PINK, font_size=11)
        + _arrow(470, 165, 470, 190)
        + _box(370, 190, 200, 45, ["Feed-Forward"], **BLUE, font_size=11)
        + _arrow(240, 140, 370, 140)
        + _label(305, 130, "K, V", size=11, color=PINK["stroke"], weight="bold")
        + _arrow(20, 200, 20, 90)
        + _label(20, 210, "Input tokens", size=10)
        + _arrow(470, 235, 470, 260)
        + _box(370, 260, 200, 30, ["Linear + Softmax -> next-token probabilities"], **GRAY, font_size=10)
        + _label(470, 285, "Output tokens, generated one at a time", size=10)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 7/8/9. BERT / GPT / T5
# ---------------------------------------------------------------------------

def bert_diagram():
    w, h = 500, 220
    inner = (
        _label(w / 2, 20, "Encoder-only: every token attends to every other token", size=12, weight="bold", color="#374151")
        + _box(20, 50, 90, 34, ["[CLS]"], **GRAY, font_size=11)
        + _box(120, 50, 90, 34, ["the"], **GRAY, font_size=11)
        + _box(220, 50, 90, 34, ["cat"], **GRAY, font_size=11)
        + _box(320, 50, 90, 34, ["[MASK]"], **AMBER, font_size=11)
        + _box(90, 110, 320, 45, ["Bidirectional self-attention stack", "(N encoder blocks)"], **PURPLE, font_size=11)
        + _arrow(65, 84, 200, 110)
        + _arrow(165, 84, 220, 110)
        + _arrow(265, 84, 240, 110)
        + _arrow(365, 84, 280, 110)
        + _arrow(250, 155, 250, 180)
        + _box(150, 180, 200, 34, ["predict masked word"], **GRAY, font_size=11)
        + _label(w / 2, 210, "Trained to fill in blanks; good at understanding, not generation", size=11)
    )
    return _wrap(inner, w, h)


def gpt_diagram():
    w, h = 500, 220
    inner = (
        _label(w / 2, 20, "Decoder-only: each token attends only to earlier tokens", size=12, weight="bold", color="#374151")
        + _box(20, 50, 90, 34, ["the"], **GRAY, font_size=11)
        + _box(120, 50, 90, 34, ["cat"], **GRAY, font_size=11)
        + _box(220, 50, 90, 34, ["sat"], **GRAY, font_size=11)
        + _box(320, 50, 90, 34, ["?"], **AMBER, font_size=11)
        + _box(90, 110, 320, 45, ["Causal (masked) self-attention stack", "(N decoder blocks, no cross-attention)"], **PURPLE, font_size=11)
        + _arrow(65, 84, 150, 110)
        + _arrow(165, 84, 200, 110)
        + _arrow(265, 84, 260, 110)
        + _arrow(365, 84, 320, 110)
        + _arrow(365, 155, 365, 180)
        + _box(270, 180, 180, 34, ["predict next word"], **GRAY, font_size=11)
        + _label(w / 2, 210, "Trained to predict the next token; generation is repeating this one step at a time", size=11)
    )
    return _wrap(inner, w, h)


def t5_diagram():
    w, h = 620, 230
    inner = (
        _label(w / 2, 20, 'Encoder-decoder, every task framed as "text in -> text out"', size=12, weight="bold", color="#374151")
        + _box(20, 50, 220, 34, ['"translate English to', 'German: That is good."'], **GRAY, font_size=10)
        + _arrow(240, 67, 290, 67)
        + _box(290, 50, 130, 50, ["Encoder", "(bidirectional)"], **PURPLE, font_size=11)
        + _arrow(420, 75, 470, 75)
        + _box(470, 50, 130, 50, ["Decoder", "(causal + cross-attn)"], **PINK, font_size=11)
        + _arrow(535, 100, 535, 140)
        + _box(440, 140, 190, 34, ['"Das ist gut."'], **GRAY, font_size=11)
        + _label(w / 2, 200, "Same architecture as the original transformer; the shift is framing every task as text-to-text", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 10. RoPE
# ---------------------------------------------------------------------------

def rope_diagram():
    w, h = 560, 260

    def vec(cx, cy, angle_deg, color, r=45):
        import math
        rad = math.radians(angle_deg)
        x2 = cx + r * math.cos(rad)
        y2 = cy - r * math.sin(rad)
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#D1D5DB" stroke-width="1"/>'
            + _arrow(cx, cy, x2, y2, color=color, marker="arrow-purple")
        )

    inner = (
        _label(140, 25, "Position 1: small rotation", size=12, weight="bold", color="#374151")
        + vec(140, 130, 20, PURPLE["fill"])
        + _label(420, 25, "Position 5: bigger rotation", size=12, weight="bold", color="#374151")
        + vec(420, 130, 80, PURPLE["fill"])
        + _label(w / 2, 210, "Same query/key vector, rotated by an angle proportional to its position", size=12)
        + _label(w / 2, 232, "Applied inside every layer's attention, not added once at the input (Su et al., 2021)", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 11. MHA vs MQA vs GQA
# ---------------------------------------------------------------------------

def gqa_diagram():
    w, h = 660, 210
    def cluster(x0, title, q_count, kv_count, share_map):
        parts = [_label(x0 + 70, 20, title, size=12, weight="bold", color="#374151")]
        q_xs = [x0 + i * 40 for i in range(q_count)]
        for i, qx in enumerate(q_xs):
            parts.append(_box(qx, 45, 32, 28, [f"Q{i+1}"], **PURPLE, font_size=9, rx=5))
        kv_xs = [x0 + 20 + i * ((q_count * 40 - 40) / max(kv_count - 1, 1) if kv_count > 1 else 0) for i in range(kv_count)]
        if kv_count == 1:
            kv_xs = [x0 + (q_count * 40 - 40) / 2]
        for j, kx in enumerate(kv_xs):
            parts.append(_box(kx, 120, 32, 28, [f"KV{j+1}"], **BLUE, font_size=9, rx=5))
        for i, qx in enumerate(q_xs):
            j = share_map[i]
            kx = kv_xs[j]
            parts.append(_arrow(qx + 16, 73, kx + 16, 120, color=ARROW_COLOR))
        return "".join(parts)

    inner = (
        cluster(10, "Multi-Head (MHA)", 4, 4, {0: 0, 1: 1, 2: 2, 3: 3})
        + cluster(230, "Grouped-Query (GQA)", 4, 2, {0: 0, 1: 0, 2: 1, 3: 1})
        + cluster(460, "Multi-Query (MQA)", 4, 1, {0: 0, 1: 0, 2: 0, 3: 0})
        + _label(w / 2, 175, "Fewer K/V heads to cache = smaller KV cache, faster generation, some quality given up", size=11)
        + _label(w / 2, 197, "GQA (Ainslie et al., 2023) is the middle ground most modern models pick", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 12. FlashAttention
# ---------------------------------------------------------------------------

def flashattention_diagram():
    w, h = 620, 260
    inner = (
        _label(150, 25, "Standard attention", size=12, weight="bold", color="#374151")
        + _box(30, 50, 240, 50, ["Compute full N x N score matrix,", "write it to slow HBM memory"], **GRAY, font_size=10)
        + _arrow(150, 100, 150, 130)
        + _box(30, 130, 240, 40, ["Read it back for softmax", "and the weighted sum"], **GRAY, font_size=10)
        + _label(150, 190, "Bottleneck: memory traffic,", size=11)
        + _label(150, 205, "not the matrix multiply itself", size=11)
        + _label(470, 25, "FlashAttention", size=12, weight="bold", color="#374151")
        + _box(350, 50, 240, 90, ["Process Q, K, V in small tiles", "that fit in fast on-chip SRAM;", "never write the full N x N", "matrix to slow memory"], **AMBER, font_size=10)
        + _label(470, 165, "Same math, same output,", size=11)
        + _label(470, 180, "far less memory traffic", size=11)
        + _label(470, 195, "(Dao et al., 2022)", size=11)
        + _label(w / 2, 235, "Attention is memory-bandwidth-bound, not compute-bound; this is an I/O optimization, not a new mechanism", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 13. Mixture of Experts
# ---------------------------------------------------------------------------

def moe_diagram():
    w, h = 620, 260
    inner = (
        _box(20, 100, 100, 40, ["Token"], **GRAY, font_size=11)
        + _arrow(120, 120, 190, 120)
        + _box(190, 100, 110, 40, ["Router", "(small gate)"], **AMBER, font_size=11)
        + _arrow(300, 110, 360, 50)
        + _arrow(300, 130, 360, 210)
        + _box(360, 20, 90, 40, ["Expert 1", "(FFN)"], **PURPLE, font_size=10)
        + _box(360, 70, 90, 40, ["Expert 2", "(FFN)"], **GRAY, font_size=10)
        + _label(405, 135, "...", size=14)
        + _box(360, 190, 90, 40, ["Expert N", "(FFN)"], **PURPLE, font_size=10)
        + _arrow(450, 40, 520, 100, dashed=True)
        + _arrow(450, 210, 520, 130, dashed=True)
        + _box(520, 95, 90, 40, ["Weighted", "sum"], **GRAY, font_size=10)
        + _arrow(610, 115, 615, 115)
        + _label(w / 2, 20, "Only the top-k experts the router picks run for this token; the rest sit idle", size=12)
        + _label(w / 2, 240, "More total parameters than a dense model, but only a slice active per token (Shazeer et al., 2017)", size=11)
    )
    return _wrap(inner, w, h)


# ---------------------------------------------------------------------------
# 14. DeepSeek: multi-head latent attention + DeepSeekMoE
# ---------------------------------------------------------------------------

def deepseek_diagram():
    w, h = 680, 320
    inner = (
        _label(150, 25, "Multi-head Latent Attention", size=12, weight="bold", color="#374151")
        + _box(20, 50, 100, 34, ["Token"], **GRAY, font_size=10)
        + _arrow(120, 67, 170, 67)
        + _box(170, 50, 150, 34, ["Compress to a", "small latent vector"], **AMBER, font_size=9)
        + _arrow(320, 67, 340, 40)
        + _arrow(320, 67, 340, 95)
        + _box(340, 20, 90, 34, ["Decompress", "-> K"], **BLUE, font_size=9)
        + _box(340, 75, 90, 34, ["Decompress", "-> V"], **BLUE, font_size=9)
        + _label(150, 130, "Only the small latent vector is KV-cached, not full K/V per head", size=10)
        + _label(150, 145, "(DeepSeek-V2, 2024)", size=10)
        + _label(540, 25, "DeepSeekMoE", size=12, weight="bold", color="#374151")
        + _box(460, 50, 100, 34, ["Shared", "expert(s)"], **PURPLE, font_size=9)
        + _label(510, 100, "always runs,", size=9)
        + _label(510, 113, "every token", size=9)
        + _box(580, 50, 100, 34, ["Many small", "routed experts"], **GRAY, font_size=9)
        + _label(630, 100, "router picks", size=9)
        + _label(630, 113, "a few, per token", size=9)
        + _arrow(20, 200, 650, 200, color="#D1D5DB")
        + _label(w / 2, 230, "Shared experts capture common patterns every token needs;", size=11)
        + _label(w / 2, 246, "routed experts specialize, more of them but each one smaller than a typical MoE expert", size=11)
        + _label(w / 2, 275, "Together: a smaller KV cache per token and more total specialization per parameter spent", size=11)
        + _label(w / 2, 295, "than the original dense transformer or a coarse-grained MoE", size=11)
    )
    return _wrap(inner, w, h)


DIAGRAMS = {
    "tokenization_embedding": tokenization_embedding,
    "scaled_dot_product_attention": scaled_dot_product_attention,
    "multi_head_attention": multi_head_attention,
    "encoder_block": encoder_block,
    "decoder_block": decoder_block,
    "full_architecture": full_architecture,
    "bert": bert_diagram,
    "gpt": gpt_diagram,
    "t5": t5_diagram,
    "rope": rope_diagram,
    "gqa": gqa_diagram,
    "flashattention": flashattention_diagram,
    "moe": moe_diagram,
    "deepseek": deepseek_diagram,
}


def render(diagram_key: str) -> str:
    return DIAGRAMS[diagram_key]()
