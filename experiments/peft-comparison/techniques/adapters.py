import model_config as mc

BOTTLENECK_DIM = 64  # common default from Houlsby et al., 2019
ADAPTERS_PER_LAYER = 2  # one after attention, one after the MLP block


def trainable_params_for_bottleneck(bottleneck_dim):
    down_proj = mc.HIDDEN_SIZE * bottleneck_dim
    up_proj = bottleneck_dim * mc.HIDDEN_SIZE
    biases = mc.HIDDEN_SIZE + bottleneck_dim
    per_adapter = down_proj + up_proj + biases
    return per_adapter * ADAPTERS_PER_LAYER * mc.NUM_LAYERS


def compute(bottleneck_dim=BOTTLENECK_DIM):
    total = mc.total_params()
    trainable = trainable_params_for_bottleneck(bottleneck_dim)
    frozen = total - trainable
    return {
        "technique": "Adapters",
        "family": "Bottleneck module",
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100 * trainable / total,
        "memory_gb": mc.memory_gb(trainable, frozen),
        "config": {"bottleneck_dim": bottleneck_dim, "adapters_per_layer": ADAPTERS_PER_LAYER},
        "mechanism": (
            f"Base weights are frozen. A small bottleneck module (down-project to "
            f"{bottleneck_dim} dimensions, nonlinearity, up-project back) is inserted "
            "in series after the attention block and after the MLP block, in every "
            "layer. Unlike LoRA, these modules sit on the model's main path, so they "
            "add a small amount of inference latency that LoRA avoids by merging back "
            "into the frozen weights after training."
        ),
        "why": (
            "One of the earliest PEFT techniques, Houlsby et al., 2019, predating "
            "LoRA. Built for the same underlying problem, adapting a large pretrained "
            "model to many tasks without storing a full copy per task."
        ),
        "tradeoffs": (
            "Roughly 10x more trainable parameters than LoRA's default settings on "
            "this model, and unlike LoRA's update, an adapter can't be merged back "
            "into the frozen weights, so it adds a small permanent cost to every "
            "inference call, not just to training. This is part of why LoRA became "
            "the more popular default over time."
        ),
        "diagram": "adapters",
    }
