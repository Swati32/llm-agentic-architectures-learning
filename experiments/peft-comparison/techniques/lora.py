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
            f"Base weights are frozen, and for each targeted matrix (query and value "
            f"projections, in every layer) a low-rank update is learned as two small "
            f"matrices, A and B (rank {rank}), added alongside the frozen weight at "
            f"inference time instead of modifying it directly. Which matrices get this "
            f"treatment is a config choice, not something LoRA works out on its own: "
            f"this experiment targets query and value projections because that's the "
            f"minimal set Hu et al.'s original paper found sufficient, not a property "
            f"the algorithm discovered about this particular model. What A and B "
            f"actually learn to contain, though, is found the ordinary way, by "
            f"gradient descent on the training data, exactly like full fine-tuning, "
            f"just restricted to a much smaller set of numbers.",
            "Hu et al., 2021, built LoRA on the bet that the change a model needs for "
            "a new task lives in a much smaller space than its full parameter count "
            "([Aghajanyan et al., 2020](https://arxiv.org/abs/2012.13255)). It's very "
            "cheap to train, and after training, A and B merge directly into the "
            "frozen weight, so a LoRA-tuned model runs at exactly the same speed as "
            "the original; the trade-off is rank, since too low a rank may not leave "
            "room to capture everything full fine-tuning could. It's the default "
            "choice for adapting an existing model cheaply whenever the base weights "
            "already fit comfortably in memory, especially when inference speed "
            "matters and it's useful to keep several task-specific variants of the "
            "same base model as small adapter files instead of full copies.",
        ],
        "diagram": "lora",
    }
