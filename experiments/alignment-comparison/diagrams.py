"""Hand-built inline SVG diagrams for the dashboard's Architecture Deep
Dive tab, one per technique plus one for LoRA. No plotting library needed.

Same visual grammar as peft-comparison's diagrams (frozen = gray,
trainable = deep purple), so the two dashboards read as one family, but
built locally rather than imported: each experiment in this repo deploys
standalone, so its dashboard can't depend on another experiment's files.
"""

FROZEN = {"fill": "#E5E7EB", "stroke": "#9CA3AF", "text_color": "#374151"}
TRAINABLE = {"fill": "#5B21B6", "stroke": "#4C1D95", "text_color": "#FFFFFF"}
SIGNAL = {"fill": "#FDF2F8", "stroke": "#DB2777", "text_color": "#9D174D"}
NEUTRAL = {"fill": "#F3F4F6", "stroke": "#D1D5DB", "text_color": "#374151"}

CANVAS_W = 700


def _box(x, y, w, h, lines, fill, stroke, text_color, font_size=12.5):
    cx = x + w / 2
    cy = y + h / 2 - (len(lines) - 1) * 8
    tspans = "".join(
        f'<tspan x="{cx}" dy="{0 if i == 0 else 15}">{line}</tspan>' for i, line in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2"/>'
        f'<text x="{cx}" y="{cy}" text-anchor="middle" font-size="{font_size}" '
        f'fill="{text_color}" font-family="sans-serif">{tspans}</text>'
    )


def _arrow(x1, y1, x2, y2, dashed=False, color="#6B7280"):
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2" marker-end="url(#arrow)"{dash}/>'
    )


def _label(x, y, text, size=11, color="#6B7280", anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-family="sans-serif">{text}</text>'


def _wrap(inner, height):
    return (
        f'<svg viewBox="0 0 {CANVAS_W} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{CANVAS_W}px;display:block;margin:0.5rem auto;">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#6B7280"/></marker></defs>'
        f"{inner}</svg>"
    )


def lora():
    parts = [
        _box(40, 70, 190, 70, ["Frozen weight W", "(unchanged)"], **FROZEN),
        _label(135, 60, "every other parameter in the model"),
        _box(40, 180, 190, 60, ["Trainable A, B", "(rank 8, this experiment)"], **TRAINABLE),
        _arrow(230, 105, 320, 105),
        _arrow(230, 210, 320, 130, dashed=True),
        _label(275, 165, "added at", size=10),
        _label(275, 178, "inference time", size=10),
        _box(320, 75, 180, 60, ["Combined output", "W(x) + B(A(x))"], **NEUTRAL),
        _label(545, 105, "Same forward pass", size=11, anchor="start"),
        _label(545, 122, "every technique in this", size=11, anchor="start"),
        _label(545, 139, "experiment builds on", size=11, anchor="start"),
    ]
    return _wrap("".join(parts), 270)


def sft():
    parts = [
        _box(20, 90, 150, 60, ["Prompt +", "chosen response"], **NEUTRAL),
        _arrow(170, 120, 230, 120),
        _box(230, 60, 170, 120, ["Base model", "(frozen)", "+", "LoRA adapter", "(trainable)"], **TRAINABLE, font_size=12),
        _arrow(400, 120, 460, 120),
        _box(460, 90, 190, 60, ["Predicted next token,", "at every position"], **NEUTRAL),
        _arrow(555, 150, 555, 195),
        _box(400, 195, 250, 55, ["Loss: how far off from the", "chosen response's actual tokens"], **SIGNAL, font_size=11.5),
        _arrow(400, 222, 315, 150, color="#DB2777"),
        _label(360, 40, "No preference pair used. No comparison to a rejected response.", size=11.5),
    ]
    return _wrap("".join(parts), 280)


def rlhf_ppo():
    parts = [
        _label(350, 22, "Stage 1: train the reward model (once)", size=12.5, color="#374151"),
        _box(20, 40, 210, 55, ["Chosen / rejected pairs", "(same pairs DPO uses)"], **NEUTRAL, font_size=11.5),
        _arrow(230, 67, 290, 67),
        _box(290, 35, 180, 65, ["Reward model", "(trained to score", "chosen > rejected)"], **TRAINABLE, font_size=11.5),
        _label(350, 128, "Stage 2: PPO optimizes the policy against it", size=12.5, color="#374151"),
        _box(20, 150, 150, 55, ["Prompt"], **NEUTRAL),
        _arrow(170, 177, 230, 177),
        _box(230, 145, 170, 65, ["Policy generates", "its own response", "(base + fresh LoRA)"], **TRAINABLE, font_size=11.5),
        _arrow(400, 177, 460, 177),
        _box(460, 150, 200, 55, ["Reward model scores", "the generated response"], **SIGNAL, font_size=11.5),
        _arrow(555, 205, 555, 240),
        _box(400, 240, 260, 55, ["PPO updates the policy toward higher-", "scoring responses, minus a KL penalty"], **SIGNAL, font_size=11),
        _arrow(400, 267, 320, 210, color="#DB2777"),
        _label(150, 320, "The KL penalty pulls back toward the frozen SFT policy whenever PPO", size=11),
        _label(150, 335, "would drift too far chasing reward, which is what mean_kl_from_reference tracks.", size=11),
    ]
    return _wrap("".join(parts), 350)


def dpo():
    parts = [
        _box(20, 30, 180, 55, ["Chosen / rejected pairs", "(same pairs RLHF uses)"], **NEUTRAL, font_size=11.5),
        _arrow(200, 57, 250, 57),
        _box(250, 20, 190, 70, ["Policy", "(base + fresh LoRA)"], **TRAINABLE, font_size=12),
        _arrow(440, 57, 500, 57),
        _box(500, 25, 180, 60, ["log-probability of chosen", "and of rejected"], **NEUTRAL, font_size=11),
        _label(345, 115, "compared against the same pair's log-probability under", size=11),
        _label(345, 130, "the frozen reference copy (the SFT policy, adapter disabled)", size=11),
        _arrow(345, 140, 345, 175, color="#DB2777"),
        _box(220, 175, 260, 60, ["DPO loss: push the gap between chosen", "and rejected further apart than the reference's"], **SIGNAL, font_size=11),
        _label(345, 260, "No response is ever generated during DPO training.", size=11.5, color="#374151"),
        _label(345, 277, "This whole step is a forward pass over existing text, not a rollout.", size=11.5, color="#374151"),
    ]
    return _wrap("".join(parts), 300)


DIAGRAMS = {"lora": lora, "sft": sft, "rlhf_ppo": rlhf_ppo, "dpo": dpo}


def render(diagram_key: str) -> str:
    return DIAGRAMS[diagram_key]()
