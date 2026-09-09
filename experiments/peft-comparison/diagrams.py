"""
Hand-built inline SVG diagrams for the Technique Deep-Dive tab, one per
technique family. No plotting library needed, and no new dependency for
the dashboard: every diagram is a plain string of SVG markup.

All six share one visual grammar so they read as a set: a frozen block
is gray, a trainable block is deep purple, and the reader can tell what
each technique trains at a glance by where the purple shows up.
"""

FROZEN = {"fill": "#E5E7EB", "stroke": "#9CA3AF", "text_color": "#374151"}
QUANTIZED = {"fill": "#D1D5DB", "stroke": "#6B7280", "text_color": "#1F2937"}
TRAINABLE = {"fill": "#5B21B6", "stroke": "#4C1D95", "text_color": "#FFFFFF"}

CANVAS_W = 640
CANVAS_H = 210


def _box(x, y, w, h, lines, fill, stroke, text_color, font_size=13):
    cx = x + w / 2
    cy = y + h / 2 - (len(lines) - 1) * 8
    tspans = "".join(
        f'<tspan x="{cx}" dy="{0 if i == 0 else 16}">{line}</tspan>' for i, line in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2"/>'
        f'<text x="{cx}" y="{cy}" text-anchor="middle" font-size="{font_size}" '
        f'fill="{text_color}" font-family="sans-serif">{tspans}</text>'
    )


def _arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#6B7280" '
        f'stroke-width="2" marker-end="url(#arrow)"{dash}/>'
    )


def _label(x, y, text, size=11, color="#6B7280"):
    return f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" fill="{color}" font-family="sans-serif">{text}</text>'


def _wrap(inner):
    return (
        f'<svg viewBox="0 0 {CANVAS_W} {CANVAS_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:640px;display:block;margin:0.5rem auto;">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#6B7280"/></marker></defs>'
        f"{inner}</svg>"
    )


def _main_row(box1_lines, box1_style, box2_lines, box2_style, caption):
    return (
        _label(20, 100, "Input")
        + _arrow(20, 105, 100, 105)
        + _box(100, 65, 170, 70, box1_lines, **box1_style)
        + _arrow(270, 105, 370, 105)
        + _box(370, 65, 170, 70, box2_lines, **box2_style)
        + _arrow(540, 105, 610, 105)
        + _label(610, 100, "Output")
        + _label(CANVAS_W / 2, 200, caption, size=12)
    )


def full_finetuning():
    return _wrap(
        _main_row(
            ["Attention", "(trainable)"], TRAINABLE,
            ["MLP", "(trainable)"], TRAINABLE,
            "Every weight, every layer: 8.03B parameters, 100% trainable",
        )
    )


def _lora_like(box1_style, box1_sublabel, box2_style, box2_sublabel, caption):
    inner = _main_row(
        ["Attention", box1_sublabel], box1_style,
        ["MLP", box2_sublabel], box2_style,
        caption,
    )
    inner += _box(150, 10, 130, 38, ["A · B", "(rank r)"], **TRAINABLE, font_size=12)
    inner += _arrow(215, 48, 215, 65, dashed=True)
    return _wrap(inner)


def lora():
    return _lora_like(
        FROZEN, "(frozen)", FROZEN, "(untouched)",
        "Only query & value projections get a rank-r update; the MLP is not targeted",
    )


def qlora():
    return _lora_like(
        QUANTIZED, "(frozen, 4-bit)", QUANTIZED, "(frozen, 4-bit)",
        "Same rank-r update as LoRA; every frozen weight, both boxes, stored in 4-bit",
    )


def adapters():
    inner = (
        _label(15, 100, "Input")
        + _arrow(15, 105, 50, 105)
        + _box(50, 65, 130, 70, ["Attention", "(frozen)"], **FROZEN)
        + _arrow(180, 105, 195, 105)
        + _box(195, 72, 90, 56, ["Adapter", "(trainable)"], **TRAINABLE, font_size=11)
        + _arrow(285, 105, 300, 105)
        + _box(300, 65, 130, 70, ["MLP", "(frozen)"], **FROZEN)
        + _arrow(430, 105, 445, 105)
        + _box(445, 72, 90, 56, ["Adapter", "(trainable)"], **TRAINABLE, font_size=11)
        + _arrow(535, 105, 600, 105)
        + _label(600, 100, "Output")
        + _label(CANVAS_W / 2, 200, "Adapters sit in series on the main path; they can't be merged away like LoRA's branch", size=12)
    )
    return _wrap(inner)


def prefix_tuning():
    inner = _main_row(
        ["Attention", "(frozen)"], FROZEN,
        ["MLP", "(frozen)"], FROZEN,
        "Virtual key/value vectors are prepended at every layer's attention; no weights touched",
    )
    inner += _box(120, 10, 190, 38, ["Virtual K/V", "(trainable, prepended)"], **TRAINABLE, font_size=11)
    inner += _arrow(185, 48, 185, 65, dashed=True)
    return _wrap(inner)


def bitfit():
    inner = _main_row(
        ["Attention", "(frozen)"], FROZEN,
        ["MLP", "(frozen)"], FROZEN,
        "Would train only bias terms, if any existed",
    )
    inner += (
        f'<rect x="230" y="150" width="180" height="34" rx="8" fill="none" '
        f'stroke="#9CA3AF" stroke-width="2" stroke-dasharray="4,4"/>'
        + _label(320, 171, "bias terms: none on llama3.1:8b", size=11, color="#6B7280")
    )
    return _wrap(inner)


DIAGRAMS = {
    "full_finetuning": full_finetuning,
    "lora": lora,
    "qlora": qlora,
    "adapters": adapters,
    "prefix_tuning": prefix_tuning,
    "bitfit": bitfit,
}


def render(diagram_key: str) -> str:
    return DIAGRAMS[diagram_key]()
