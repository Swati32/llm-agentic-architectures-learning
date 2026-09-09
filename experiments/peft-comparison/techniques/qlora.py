import model_config as mc
from techniques.lora import trainable_params_for_rank, TARGET_MODULES

QUANTIZED_WEIGHT_BYTES = 0.5  # 4-bit NF4


def compute(rank=8):
    total = mc.total_params()
    trainable = trainable_params_for_rank(rank)  # identical to LoRA, on purpose
    frozen = total - trainable
    return {
        "technique": "QLoRA",
        "family": "Reparameterization",
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100 * trainable / total,
        "memory_gb": mc.memory_gb(trainable, frozen, base_weight_bytes=QUANTIZED_WEIGHT_BYTES),
        "config": {"rank": rank, "target_modules": TARGET_MODULES, "quantization": "4-bit NF4"},
        "mechanism": (
            f"Exactly LoRA's mechanism (rank {rank} update on query and value "
            "projections), with one change: the frozen base weights are stored in "
            "4-bit instead of 16-bit. Trainable parameter count is identical to LoRA. "
            "Only the frozen weight storage shrinks."
        ),
        "why": (
            "Built by Dettmers et al., 2023, to fit LoRA fine-tuning of much bigger "
            "models onto a single consumer GPU, by shrinking the frozen weights "
            "rather than changing what gets trained."
        ),
        "tradeoffs": (
            "Same LoRA update, a quarter of LoRA's frozen-weight memory. Some "
            "precision is lost by storing weights in 4-bit, though the paper's "
            "specific quantization scheme (double quantization + the NF4 data type) "
            "is designed to keep that loss small enough not to show up in quality."
        ),
        "when_to_use": (
            "Reach for this specifically when the base model doesn't fit in available "
            "GPU memory at 16-bit at all, for example fine-tuning a 70B model on a "
            "single consumer GPU. If the model already fits comfortably, plain LoRA "
            "avoids QLoRA's small quantization-precision cost for no real benefit."
        ),
        "deep_dive": [
            "Exactly LoRA's mechanism, and exactly LoRA's answer to \"which "
            "parameters\": the same config choice (query and value projections, rank "
            f"{rank}) decides where the trainable A and B matrices go, and gradient "
            "descent decides what they contain. The only change is to the frozen "
            "weights, which are stored in 4-bit instead of 16-bit; that's a storage "
            "decision applied uniformly to every frozen parameter, not a selection "
            "among them.",
            "Dettmers et al., 2023, built QLoRA to fit LoRA fine-tuning of much "
            "bigger models onto a single consumer GPU, by shrinking the frozen "
            "weights rather than changing what gets trained. It gets LoRA's exact "
            "update at a quarter of LoRA's frozen-weight memory; some precision is "
            "lost by storing weights in 4-bit, though the paper's specific "
            "quantization scheme (double quantization plus the NF4 data type) is "
            "designed to keep that loss small enough not to show up in quality. "
            "Reach for this specifically when the base model doesn't fit in "
            "available GPU memory at 16-bit at all; if it already fits comfortably, "
            "plain LoRA avoids this small precision cost for no real benefit.",
        ],
        "diagram": "qlora",
    }
