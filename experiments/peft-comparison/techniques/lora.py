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
        "when_to_use": (
            "The default choice for adapting an existing model cheaply, when the base "
            "weights already fit comfortably in memory. Especially good when inference "
            "speed matters, since the merged model runs exactly as fast as the "
            "original, and when you want to keep several task-specific variants of the "
            "same base model around as small adapter files instead of full copies."
        ),
        "deep_dive": [
            "Instead of letting every number in the model change, LoRA leaves "
            "almost all of them exactly as they are, frozen, and adds a small side "
            "path next to a few chosen spots, here, the parts of each layer that "
            "decide what the model should pay attention to. That side path is "
            f"built from two small tables of numbers, nicknamed A and B (this "
            f"experiment uses rank {rank}, meaning how big those tables are), and "
            "only those get trained. Once training is done, A and B's effect can "
            "be folded directly back into the original numbers, so the finished "
            "model runs exactly as fast as it did before, with nothing extra "
            "bolted on.",
            "Where exactly to add that side path is a decision someone made ahead "
            "of time, not something LoRA works out by studying the model. The team "
            "that invented LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) "
            "tried adding it in different spots and found that a model's attention "
            "layers alone were enough to get most of the benefit, on the theory "
            "that the change a model needs for a new task is much simpler than its "
            "full size suggests ([Aghajanyan et al., 2020](https://arxiv.org/abs/2012.13255)). "
            "What A and B actually end up containing, though, is learned the "
            "completely normal way, just by training on examples, only on a much "
            "smaller set of numbers. This is usually the first thing to try when "
            "fine-tuning a model cheaply: it barely uses extra memory, keeps the "
            "model just as fast, and lets you keep several fine-tuned versions "
            "around as small add-on files instead of full copies of the model.",
        ],
        "diagram": "lora",
    }
