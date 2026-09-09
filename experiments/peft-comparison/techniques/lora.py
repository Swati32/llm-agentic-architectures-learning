import model_config as mc

# LoRA's original paper targets only the query and value projections, the
# minimal configuration that already recovers most of full fine-tuning's
# quality. Real-world configs often extend to all four attention
# projections and the MLP matrices for a quality bump, at proportionally
# more trainable parameters; that trade-off is covered in the README rather
# than computed here, to keep this table's headline row comparable to the
# paper's own reported numbers.
TARGET_MODULES = ["q_proj", "v_proj"]


def trainable_params_for_rank(rank):
    q_proj = rank * (mc.HIDDEN_SIZE + mc.HIDDEN_SIZE)
    v_proj = rank * (mc.HIDDEN_SIZE + mc.KV_DIM)
    per_layer = q_proj + v_proj
    return per_layer * mc.NUM_LAYERS


def compute(rank=8):
    total = mc.total_params()
    trainable = trainable_params_for_rank(rank)
    frozen = total - trainable
    return {
        "technique": "LoRA",
        "family": "Reparameterization",
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100 * trainable / total,
        "memory_gb": mc.memory_gb(trainable, frozen),
        "config": {"rank": rank, "target_modules": TARGET_MODULES},
        "mechanism": (
            f"Base weights are frozen. For each targeted matrix (query and value "
            f"projections, in every layer), a low-rank update is learned as two small "
            f"matrices A and B (rank {rank}) and added alongside the frozen weight at "
            f"inference time, instead of modifying it directly."
        ),
        "why": (
            "Introduced by Hu et al., 2021, on the bet that the change a model needs "
            "for a new task lives in a much smaller space than its full parameter "
            "count (Aghajanyan et al., 2020). If that's true, you only need to learn "
            "a small update, not a full new set of weights."
        ),
        "tradeoffs": (
            "Very cheap to train, and after training, A and B can be merged directly "
            "into the frozen weight, so a LoRA-tuned model runs at exactly the same "
            "speed as the original at inference time. The trade-off is rank: too low "
            "and the update may not have room to capture everything full fine-tuning "
            "could."
        ),
        "diagram": "lora",
    }
